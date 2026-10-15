"""
05_gcn_experiment.py — Train the GCN and evaluate it on three negative sets.

This is the main experiment. One run trains a 3-layer GCN link predictor on
OGB-ddi and scores the SAME test positives against three different negative
sets. The contrast between those three numbers is the result; no single one
of them means much on its own.

    standard benchmark   ogb_test_pos vs random_neg_test
    degree-matched       ogb_test_pos vs degmatched_neg_test
    CRESCENDDI           ogb_test_pos vs cres_test

Usage
-----
    # one configuration
    python 05_gcn_experiment.py --features pubmedbert --train-neg random --seed 0

    # the full feature sweep, 5 seeds (this is what produced results/)
    python 05_gcn_experiment.py --sweep --seeds 5

Outputs, into data/results/
    gcn_{train_neg}_{features}_seed{n}.pt            encoder + decoder + config
    scores_{train_neg}_{features}_seed{n}_{split}_{pos,neg}.npy
    metrics_{train_neg}_{features}_seed{n}.json

Note that the ORIGINAL runs saved only the encoder, under the shorter name
gcn_{train_neg}_{features}.pt and with no config dict. That is why the decoder
had to be reconstructed post-hoc months later. This version saves both plus
the configuration; do not remove that.

Where to run it
---------------
Renku has no usable GPU — the driver is too old and torch silently falls back
to CPU, where 400 epochs of full-batch training on 1M edges is impractical.
Training was done on Google Colab (A100/H100), roughly 20-40 min per run.
Everything else in this project (diagnostics, baselines, scoring) runs fine
on Renku CPU. See ddi_toolkit.py for those.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GCNConv

DATA = Path("/home/renku/work/data")
FROZEN = DATA / "frozen_r5"
RESULTS = DATA / "results"

FEATURE_FILES = {
    "pubmedbert": "pubmedbert_embeddings.npy",
    "chemberta":  "chemberta_embeddings.npy",
    "gpt":        "gpt_embeddings.npy",
    "random1536": "random1536_embeddings.npy",
    "pmbwhite":   "pmbwhite_embeddings.npy",
    "cbwhite":    "cbwhite_embeddings.npy",
    "bothwhite":  "bothwhite_embeddings.npy",
}

# Same positives everywhere; only the negatives change. That is the point.
EVAL_SETS = {
    "random":         ("ogb_test_pos.npy", "random_neg_test.npy"),
    "degree_matched": ("ogb_test_pos.npy", "degmatched_neg_test.npy"),
    "crescenddi":     ("ogb_test_pos.npy", "cres_test.npy"),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# See the note in load_everything(). False matches the original runs.
SYMMETRISE = False


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """768 -> 256 -> 256 -> 256. ReLU between layers, none after the last.

    Three layers means each drug's embedding carries information from three
    hops away. In a graph at 14.7% density that is most of the graph, which
    is part of why the embeddings end up encoding little beyond degree.
    """

    def __init__(self, in_dim: int, hidden: int = 256, n_layers: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        dims = [in_dim] + [hidden] * n_layers
        self.convs = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1]) for i in range(n_layers)])
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                if self.dropout:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class Decoder(nn.Module):
    """MLP over the Hadamard product. Symmetric, as an undirected edge requires.

    Concatenating [z_i ; z_j] would NOT be symmetric and would let the model
    learn an arbitrary column order. The elementwise product is symmetric by
    construction.
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, z, pairs):
        return self.mlp(z[pairs[:, 0]] * z[pairs[:, 1]]).squeeze(-1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

N_NODES = 4267


def load_everything(feature_name: str):
    x_np = np.load(DATA / FEATURE_FILES[feature_name])
    train_pos = np.load(FROZEN / "ogb_train_pos.npy")

    # gpt_embeddings.npy has only 3,481 rows: the DDI-LLM pipeline it came from
    # filtered on description AND SMILES, covering 84.5% of OGB-ddi. Indexing it
    # with node ids up to 4,266 would either raise or, worse, silently read the
    # wrong drug. Refuse rather than guess — padding with zeros or means would
    # invent features for 786 drugs and quietly change what is being measured.
    if x_np.shape[0] != N_NODES:
        raise ValueError(
            f"{FEATURE_FILES[feature_name]} has {x_np.shape[0]} rows, "
            f"expected {N_NODES}. This feature file does not cover every OGB "
            f"node and cannot be used as-is. See DATA.md.")

    x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)

    # SYMMETRISE is False, deliberately. ogb_train_pos stores each undirected
    # edge once and GCNConv does not add the reverse, so messages flow in one
    # direction only. Unusual for an undirected graph, but it is what the
    # original runs did: reloading gcn_random_pubmedbert.pt reproduces the
    # reference mean norm (4.327) and r(norm, log-degree) (0.756) only with
    # False. Flipping it to True is a new experiment, not a bug fix — it would
    # make new runs incomparable with every stored result.
    ei = torch.tensor(train_pos.T, dtype=torch.long)
    if SYMMETRISE:
        ei = torch.cat([ei, ei.flip(0)], dim=1)
    edge_index = ei.to(DEVICE)
    return x, edge_index, train_pos


def training_pairs(train_neg: str, rng, subsample_ratio: int = 10,
                   max_pairs: int = None):
    """Positives and negatives for one training regime.

    random:     1,066,276 positives vs 1,066,276 uniform non-edges, 1:1.
    crescenddi: only 2,834 verified negatives exist, so positives are
                subsampled to 10:1 and pos_weight compensates the rest.
                Upsampling the negatives instead was tried and hurt
                validation — weight the loss, do not resample the data.
    """
    pos = np.load(FROZEN / "ogb_train_pos.npy")
    if train_neg == "random":
        neg = np.load(FROZEN / "random_neg_train.npy")
        pos_weight = 1.0
    elif train_neg == "crescenddi":
        neg = np.load(FROZEN / "cres_train.npy")
        keep = min(len(pos), len(neg) * subsample_ratio)
        pos = pos[rng.choice(len(pos), keep, replace=False)]
        pos_weight = 0.1
    else:
        raise ValueError(train_neg)

    # CPU validation escape hatch. Full-batch on 2.1M pairs needs a GPU; with
    # max_pairs=50_000 an epoch takes seconds on CPU. The absolute AUROCs come
    # out lower, but the ORDERING of the three evaluation sets and the size of
    # the spread between them are preserved — which is what the experiment is
    # about. Never report numbers from a subsampled run.
    if max_pairs:
        if len(pos) > max_pairs:
            pos = pos[rng.choice(len(pos), max_pairs, replace=False)]
        if len(neg) > max_pairs:
            neg = neg[rng.choice(len(neg), max_pairs, replace=False)]

    return pos, neg, pos_weight


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_pairs(decoder, z, pairs, batch: int = 200_000):
    """Score pairs against an ALREADY-COMPUTED z. Batched to bound memory.

    z is passed in rather than recomputed: the encoder forward over the whole
    graph is the expensive part, and it is identical for every evaluation set.
    """
    decoder.eval()
    out = []
    for i in range(0, len(pairs), batch):
        p = torch.tensor(pairs[i:i + batch], dtype=torch.long, device=DEVICE)
        out.append(decoder(z, p).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def evaluate(encoder, decoder, x, edge_index, save_tag: str = None) -> dict:
    """AUROC on all three evaluation sets. Saves raw scores if given a tag.

    All three sets share the same positives (ogb_test_pos), so those 133,117
    pairs are scored ONCE and reused. One encoder forward, one positive pass,
    three negative passes — instead of six full passes.

    Save the scores. The decoder weights of the original runs were not kept,
    and these .npy files were the only way to recover the learned-decoder
    column months later without retraining.
    """
    encoder.eval(); decoder.eval()
    z = encoder(x, edge_index)

    pos_files = {fp for fp, _ in EVAL_SETS.values()}
    pos_cache = {fp: score_pairs(decoder, z, np.load(FROZEN / fp))
                 for fp in pos_files}

    metrics = {}
    for name, (fp, fn) in EVAL_SETS.items():
        pos = pos_cache[fp]
        neg = score_pairs(decoder, z, np.load(FROZEN / fn))
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        metrics[name] = float(roc_auc_score(y, np.r_[pos, neg]))
        if save_tag:
            RESULTS.mkdir(parents=True, exist_ok=True)
            np.save(RESULTS / f"scores_{save_tag}_{name}_pos.npy", pos)
            np.save(RESULTS / f"scores_{save_tag}_{name}_neg.npy", neg)
    metrics["spread"] = max(metrics[k] for k in EVAL_SETS) - \
                        min(metrics[k] for k in EVAL_SETS)
    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run(features: str, train_neg: str, seed: int, epochs: int = 400,
        lr: float = 5e-4, hidden: int = 256, verbose: bool = True,
        max_pairs: int = None, save: bool = True,
        pair_batch: int = None) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    x, edge_index, _ = load_everything(features)
    pos, neg, pos_weight = training_pairs(train_neg, rng, max_pairs=max_pairs)

    pos_t = torch.tensor(pos, dtype=torch.long, device=DEVICE)
    neg_t = torch.tensor(neg, dtype=torch.long, device=DEVICE)
    y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))]).to(DEVICE)
    pairs = torch.cat([pos_t, neg_t])

    encoder = Encoder(x.size(1), hidden).to(DEVICE)
    decoder = Decoder(hidden).to(DEVICE)
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=DEVICE))

    # Memory note. Scoring all 2.1M pairs in one go materialises a
    # [2.1M, 256] activation per MLP layer — about 2 GB each in float32, and
    # autograd keeps them for the backward pass. Peak is roughly 7-8 GB, fine
    # on an A100 but not on a 16 GB card. pair_batch splits the DECODER pass
    # into chunks while keeping the encoder full-batch, so the graph the model
    # sees is unchanged; gradients accumulate and the update is identical.
    params = list(encoder.parameters()) + list(decoder.parameters())
    n_pairs = len(pairs)

    for epoch in range(1, epochs + 1):
        encoder.train(); decoder.train()
        opt.zero_grad()
        z = encoder(x, edge_index)

        if pair_batch is None or pair_batch >= n_pairs:
            loss = loss_fn(decoder(z, pairs), y)
            loss.backward()
            epoch_loss = loss.item()
        else:
            epoch_loss = 0.0
            for i in range(0, n_pairs, pair_batch):
                sl = slice(i, i + pair_batch)
                chunk = loss_fn(decoder(z, pairs[sl]), y[sl])
                # weight each chunk by its share so the total equals the
                # full-batch mean loss, and the gradient does too
                (chunk * (len(y[sl]) / n_pairs)).backward(retain_graph=True)
                epoch_loss += chunk.item() * len(y[sl]) / n_pairs

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if verbose and epoch % 50 == 0:
            print(f"    epoch {epoch:>4}  loss {epoch_loss:.4f}")

    # Report the LAST epoch, not a validation-selected one. The validation
    # AUROC saturates around epoch 15 while the control-set AUROC keeps
    # improving until ~300: the degree shortcut is there from epoch 1 and
    # takes most of training to partially resolve. Early stopping on
    # validation therefore stops exactly when the interesting part begins.
    tag = f"{train_neg}_{features}_seed{seed}"
    metrics = evaluate(encoder, decoder, x, edge_index,
                       save_tag=tag if save else None)

    if save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        torch.save({"encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "config": dict(features=features, train_neg=train_neg,
                                   seed=seed, epochs=epochs, lr=lr,
                                   hidden=hidden, pos_weight=pos_weight,
                                   max_pairs=max_pairs, pair_batch=pair_batch,
                                   n_train_pos=len(pos), n_train_neg=len(neg))},
                   RESULTS / f"gcn_{train_neg}_{features}_seed{seed}.pt")
        json.dump(metrics, open(RESULTS / f"metrics_{tag}.json", "w"), indent=2)

    if verbose:
        print(f"    " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="pubmedbert", choices=list(FEATURE_FILES))
    p.add_argument("--train-neg", default="random", choices=["random", "crescenddi"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=1, help="with --sweep")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--sweep", action="store_true",
                   help="every feature type x both regimes x --seeds seeds")
    p.add_argument("--max-train-pairs", type=int, default=None,
                   help="subsample training pairs. For CPU validation only — "
                        "results are not comparable to full runs.")
    p.add_argument("--no-save", action="store_true",
                   help="do not write checkpoints or scores (validation runs)")
    p.add_argument("--pair-batch", type=int, default=None,
                   help="chunk the decoder pass during training. Same update, "
                        "lower peak memory. Try 500000 on a 16GB card.")
    args, _ = p.parse_known_args()

    if DEVICE == "cpu" and not args.max_train_pairs:
        print("WARNING: no GPU, and no --max-train-pairs. Full-batch training "
              "on 2.1M pairs\nwill take many hours here. Either train on Colab, "
              "or pass\n--max-train-pairs 50000 --epochs 150 --no-save to "
              "validate the pipeline.\n")

    common = dict(max_pairs=args.max_train_pairs, save=not args.no_save,
                  pair_batch=args.pair_batch)

    if not args.sweep:
        run(args.features, args.train_neg, args.seed, args.epochs, args.lr,
            **common)
        return

    table, skipped = {}, []
    for train_neg in ("random", "crescenddi"):
        for feat in FEATURE_FILES:
            runs = []
            for s in range(args.seeds):
                print(f"\n[{train_neg} | {feat} | seed {s}]")
                try:
                    runs.append(run(feat, train_neg, s, args.epochs, args.lr,
                                    **common))
                except (ValueError, FileNotFoundError) as e:
                    # One unusable feature file must not take the sweep down
                    # with it — gpt_embeddings.npy covers only 3,481 nodes.
                    print(f"    SKIPPED: {e}")
                    skipped.append(f"{train_neg}|{feat}")
                    break
            if not runs:
                continue
            table[f"{train_neg}|{feat}"] = {
                k: (float(np.mean([r[k] for r in runs])),
                    float(np.std([r[k] for r in runs])))
                for k in runs[0]}

    if not table:
        print("\nNothing ran. Check the feature files and frozen_r5/.")
        return

    print("\n" + "=" * 78)
    print(f"{'config':<26}" + "".join(f"{k:>17}" for k in EVAL_SETS))
    print("=" * 78)
    for cfg, m in table.items():
        print(f"{cfg:<26}" +
              "".join(f"{m[k][0]:>11.3f}±{m[k][1]:.3f}" for k in EVAL_SETS))
    if skipped:
        print(f"\nskipped: {', '.join(skipped)}")

    if not args.no_save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        json.dump(table, open(RESULTS / "sweep_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# What to expect
# ---------------------------------------------------------------------------
#
# random | pubmedbert, 5 seeds:
#   benchmark 0.902 ± 0.010 · degree-matched ~0.66 · CRESCENDDI ~0.51
#   spread 0.417 — the model is nearly blind once degree is controlled.
#
# crescenddi | pubmedbert:
#   spread collapses to 0.092. Training on verified negatives is the single
#   change that most reduces the gap between evaluation sets.
#
# Baselines worth keeping in view:
#   logistic regression on the two endpoint log-degrees alone: 0.880 on the
#   benchmark, i.e. 97% of the GCN. Adding 1,536 embedding dimensions on top
#   buys 0.010. If a change does not move the degree-matched column, it has
#   not improved anything that matters.
#
# Raw embeddings underperform random Gaussian vectors on the control sets,
# because of anisotropy (PubMedBERT mean pairwise cosine 0.921). The whitened
# variants fix this and reverse the ranking. Run both.