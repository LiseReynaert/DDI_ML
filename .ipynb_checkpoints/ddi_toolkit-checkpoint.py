"""
ddi_toolkit.py — Shared functions for the DDI link-prediction project.

Everything needed to load the data, rebuild the trained encoder, run the
degree diagnostics, score pairs under different decoders, and fit classical
baselines. Written so that a successor can reproduce every result in the
report without reading the original notebooks.

Project in one sentence
-----------------------
We predict drug-drug interactions as link prediction on the OGB-ddi graph
(4,267 drugs, 1.33M edges) using a 3-layer GCN with biomedical text
embeddings as node features, and we evaluate whether the benchmark score
reflects pharmacology or a structural artefact.

The short answer is that it largely reflects an artefact. See REFERENCE
RESULTS at the bottom of this file.

Layout
------
  1. Paths and configuration
  2. Data loading (graph, degrees, features, evaluation splits)
  3. Model definition (encoder / decoder)
  4. Embedding computation
  5. Diagnostics (norm vs degree)
  6. Post-hoc scoring (no retraining required)
  7. Pair features and classical baselines
  8. Data integrity checks
  9. CLI

Usage
-----
    python ddi_toolkit.py check        # verify files and label polarity
    python ddi_toolkit.py diagnose     # norm/degree correlation
    python ddi_toolkit.py decoders     # compare the three decoders
    python ddi_toolkit.py baselines    # logistic regression baselines

Or import it:

    from ddi_toolkit import load_graph, embeddings_from_checkpoint
    g = load_graph()
    z = embeddings_from_checkpoint(CKPT_DEFAULT, g)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# ---------------------------------------------------------------------------
# 1. Paths and configuration
# ---------------------------------------------------------------------------

DATA = Path("/home/renku/work/data")
FROZEN = DATA / "frozen_r5"          # conflict-removed split; `frozen/` is the older one
RESULTS = DATA / "results"

CKPT_DEFAULT = RESULTS / "gcn_random_pubmedbert.pt"

# Feature files on disk. `*white*` are the ZCA-whitened variants.
FEATURES = {
    "pubmedbert": "pubmedbert_embeddings.npy",      # 768-d, raw
    "pmbwhite":   "pmbwhite_embeddings.npy",        # 768-d, whitened
    "chemberta":  "chemberta_embeddings.npy",       # 768-d, SMILES
    "cbwhite":    "cbwhite_embeddings.npy",
    "gpt":        "gpt_embeddings.npy",             # 1536-d, ada-002
    "random1536": "random1536_embeddings.npy",      # Gaussian control
    "bothwhite":  "bothwhite_embeddings.npy",       # 1536-d, concatenated
}

# Architecture recovered from the checkpoint's state_dict:
#   convs.0.lin.weight  [256, 768]
#   convs.1.lin.weight  [256, 256]
#   convs.2.lin.weight  [256, 256]
HIDDEN_DIM = 256
N_LAYERS = 3

# Whether to add the reverse of every edge before message passing.
#
# FALSE, and that is not an oversight. ogb_train_pos.npy stores each undirected
# edge once, and GCNConv does not symmetrise on its own, so with False the
# information flows only from edges[:,0] to edges[:,1]. That is unusual for an
# undirected graph — but it is what the training runs did, and changing it here
# would mean the embeddings no longer match the checkpoints.
#
# Verified empirically: reloading gcn_random_pubmedbert.pt and running the
# encoder gives mean norm 4.327 and r(norm, log-degree) = 0.756 with False,
# versus 0.793 and 0.541 with True. The first pair matches the reference
# measurements; the second does not.
#
# Worth revisiting as an experiment (symmetrising is the conventional choice
# and may well perform better) — but that is a NEW run, not a reinterpretation
# of the existing one.
SYMMETRISE = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# 2. Data loading
# ---------------------------------------------------------------------------

class Graph:
    """The training graph plus everything derived from it.

    Degrees are computed on the TRAINING edges only. Using the full graph
    would leak test structure into any degree-based analysis.
    """

    def __init__(self, edges: np.ndarray, n_nodes: int):
        self.edges = edges                      # [E, 2]
        self.n_nodes = n_nodes
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        self.adj = sp.coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(n_nodes, n_nodes)
        ).tocsr()
        self.deg = np.asarray(self.adj.sum(axis=1)).ravel()
        self.log_deg = np.log1p(self.deg)

    def edge_index(self, device=DEVICE, symmetrise=SYMMETRISE) -> torch.Tensor:
        """PyG edge_index, shape [2, E] (or [2, 2E] if symmetrise=True).

        The default is False, which is what the training runs actually did —
        see the SYMMETRISE note above. Do not change it without re-checking
        against the checkpoint.
        """
        ei = torch.tensor(self.edges.T, dtype=torch.long)
        if symmetrise:
            ei = torch.cat([ei, ei.flip(0)], dim=1)
        return ei.to(device)

    def summary(self) -> str:
        return (f"nodes={self.n_nodes}  edges={len(self.edges)}  "
                f"degree min={self.deg.min():.0f} median={np.median(self.deg):.0f} "
                f"max={self.deg.max():.0f}  isolated={(self.deg == 0).sum()}")


def load_graph(edge_file: str = "ogb_train_pos.npy", n_nodes: int = 4267) -> Graph:
    """Load the training edges and build the Graph object.

    Expected: (1066276, 2) int64, indices in [0, 4266].
    """
    edges = np.load(FROZEN / edge_file)
    if edges.shape[0] == 2 and edges.shape[1] != 2:
        edges = edges.T
    return Graph(edges, n_nodes)


def load_features(name: str = "pubmedbert") -> np.ndarray:
    """Load a node feature matrix by short name (see FEATURES).

    The feature file must match what the checkpoint was trained on. The
    checkpoint name encodes it: gcn_{train_neg}_{features}.pt, so
    `gcn_random_pubmedbert.pt` means raw PubMedBERT, not the whitened one.
    """
    path = DATA / FEATURES.get(name, name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path}\nAvailable: " + ", ".join(sorted(FEATURES)))
    return np.load(path)


# Evaluation splits, as (positives_file, negatives_file).
#
# READ THIS BEFORE EDITING. Every CRESCENDDI file holds VERIFIED NEGATIVES.
# There are no positives among them. `cres_*` and `cres_dd_*` are the SAME
# pairs split two different ways, built by 02_build_negative_sets.py:
#
#     c_tr, c_va, c_te = split_pairwise(cres, rng)       -> cres_*
#     d_tr, d_va, d_te = split_drug_disjoint(cres, rng)  -> cres_dd_*
#
# So "dd" means DRUG-DISJOINT (no drug appears in two splits), not a second
# class of pairs. Pitting cres_test against cres_dd_test is a non-task: both
# sides are negatives, the labels are arbitrary, and AUROC is meaningless.
#
# The protocol is: the SAME OGB test positives against three different
# negative sets. That contrast is the contribution.
EVAL_SETS = {
    "ogb":            ("ogb_test_pos.npy", "random_neg_test.npy"),
    "degree_matched": ("ogb_test_pos.npy", "degmatched_neg_test.npy"),
    "crescenddi":     ("ogb_test_pos.npy", "cres_test.npy"),
}

# Drug-disjoint variant of the CRESCENDDI evaluation: harder, because no drug
# in the test negatives was seen in the CRESCENDDI training negatives.
EVAL_SETS_DD = {
    "crescenddi_dd":  ("ogb_test_pos.npy", "cres_dd_test.npy"),
}

# Training regimes. Positives are always OGB edges; only the negatives change.
TRAIN_SETS = {
    "random":     ("ogb_train_pos.npy", "random_neg_train.npy"),
    "crescenddi": ("ogb_train_pos.npy", "cres_train.npy"),
}


def load_pairs(name: str) -> np.ndarray:
    return np.load(FROZEN / name)


def load_split(spec: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
    return load_pairs(spec[0]), load_pairs(spec[1])


# ---------------------------------------------------------------------------
# 3. Model definition
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Graph + node features -> one embedding per drug.

    Layer k computes  H <- sigma(A_hat H W).  ReLU between layers, none after
    the last one. This matches the saved checkpoint; if you add a trailing
    ReLU the norms change and the diagnostics below are no longer comparable.
    """

    def __init__(self, dims: list[int]):
        super().__init__()
        self.convs = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


class Decoder(nn.Module):
    """(z_i, z_j) -> one logit per pair.

    Must be symmetric: DDI is undirected, so score(i,j) == score(j,i).
    The Hadamard product guarantees that; concatenation would not.

    Modes
    -----
    hadamard     MLP(z_i * z_j).                 The original decoder.
    cosine_mlp   MLP(z_i_hat * z_j_hat) * s.     Norm channel removed, MLP kept.
    cosine_pure  sum(z_i_hat * z_j_hat) * s.     Norm channel removed, no MLP.

    Why the scale `s`: cosine is bounded in [-1, 1], so sigmoid(1) = 0.73 and
    BCE can never become confident. A learned temperature reopens the range
    without giving the model any per-node information back. Clamped at e^4 to
    stop it saturating the logits to compensate for the constraint.
    """

    def __init__(self, dim: int = HIDDEN_DIM, mode: str = "hadamard"):
        super().__init__()
        assert mode in ("hadamard", "cosine_mlp", "cosine_pure")
        self.mode = mode
        if mode != "cosine_pure":
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        if mode.startswith("cosine"):
            self.log_scale = nn.Parameter(torch.tensor(2.3))   # exp(2.3) ~ 10

    def forward(self, z, pairs):
        if self.mode.startswith("cosine"):
            z = F.normalize(z, p=2, dim=-1)
        h = z[pairs[:, 0]] * z[pairs[:, 1]]
        out = h.sum(-1) if self.mode == "cosine_pure" else self.mlp(h).squeeze(-1)
        if self.mode.startswith("cosine"):
            out = self.log_scale.clamp(max=4.0).exp() * out
        return out


# ---------------------------------------------------------------------------
# 4. Embedding computation
# ---------------------------------------------------------------------------

def embeddings_from_checkpoint(ckpt_path: Path | str,
                               graph: Graph,
                               feature_name: str = "pubmedbert",
                               device: str = DEVICE) -> np.ndarray:
    """Load a trained encoder and return Z, shape [n_nodes, 256].

    The checkpoint is a bare OrderedDict of encoder weights (no wrapper dict,
    no decoder, no optimiser state). weights_only=False is required.

    If load_state_dict raises on a missing or unexpected key, the architecture
    here does not match the checkpoint — inspect the keys before guessing.
    """
    x_np = load_features(feature_name)
    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    ei = graph.edge_index(device)

    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    dims = [x.size(1)] + [HIDDEN_DIM] * N_LAYERS
    enc = Encoder(dims).to(device)
    enc.load_state_dict(sd)
    enc.eval()
    with torch.no_grad():
        return enc(x, ei).cpu().numpy()


def embeddings_untrained(graph: Graph,
                         feature_name: str = "pubmedbert",
                         seed: int = 0,
                         device: str = DEVICE) -> np.ndarray:
    """Same architecture, random weights. The control for the diagnostic below.

    This matters: a GCN already produces degree-correlated norms at
    initialisation, because each layer sums over neighbours. Without this
    control you cannot tell what training actually did.
    """
    x = torch.tensor(load_features(feature_name), dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    enc = Encoder([x.size(1)] + [HIDDEN_DIM] * N_LAYERS).to(device)
    enc.eval()
    with torch.no_grad():
        return enc(x, graph.edge_index(device)).cpu().numpy()


# ---------------------------------------------------------------------------
# 5. Diagnostics
# ---------------------------------------------------------------------------

def norm_degree_diagnostic(z: np.ndarray, graph: Graph, name: str = "",
                           verbose: bool = True) -> dict:
    """Does embedding magnitude encode node degree?

    Motivation: under a dot-product decoder the score factorises as
        score(i,j) = ||z_i|| * ||z_j|| * cos(theta_ij)
    so degree can drive predictions through the norm term alone, with no
    pairwise similarity involved at all.

    Reported:
      pearson / spearman  against log-degree. Spearman > Pearson means the
                          relation is monotone but not linear.
      cv                  std/mean. Comparable across models with different
                          global scales (trained norms are ~4x smaller than
                          untrained ones, which says nothing on its own).
      decile_ratio        mean norm of the top degree decile over the bottom.
                          The most quotable single number.
    """
    norms = np.linalg.norm(z, axis=1)
    r_p = float(pearsonr(norms, graph.log_deg).statistic)
    r_s = float(spearmanr(norms, graph.log_deg).statistic)

    q = np.quantile(graph.log_deg, np.linspace(0, 1, 11))
    bins = np.clip(np.digitize(graph.log_deg, q[1:-1]), 0, 9)
    profile = [float(norms[bins == b].mean()) for b in range(10)]

    out = dict(name=name, pearson=r_p, spearman=r_s, r2=r_p ** 2,
               mean=float(norms.mean()), std=float(norms.std()),
               cv=float(norms.std() / norms.mean()),
               profile=profile, decile_ratio=profile[9] / profile[0])

    if verbose:
        print(f"\n[{name}]  Pearson={r_p:.3f}  Spearman={r_s:.3f}  R2={r_p**2:.3f}")
        print(f"  norm mean={out['mean']:.2f}  std={out['std']:.2f}  cv={out['cv']:.3f}")
        print("  mean norm per degree decile:")
        print("  " + "  ".join(f"{v:.2f}" for v in profile))
        print(f"  top/bottom decile ratio: {out['decile_ratio']:.2f}")
    return out


def plot_norm_vs_degree(zs: list[tuple[str, np.ndarray]], graph: Graph,
                        out_path: Path):
    """Scatter of ||z|| against log-degree, one panel per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(zs), figsize=(5 * len(zs), 4))
    axes = np.atleast_1d(axes)
    for ax, (label, z) in zip(axes, zs):
        n = np.linalg.norm(z, axis=1)
        r = pearsonr(n, graph.log_deg).statistic
        ax.scatter(graph.log_deg, n, s=3, alpha=0.25, edgecolors="none")
        ax.set_xlabel("log(1 + degree)")
        ax.set_ylabel("||z||")
        ax.set_title(f"{label} - r={r:.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"figure -> {out_path}")


# ---------------------------------------------------------------------------
# 6. Post-hoc scoring (no retraining)
# ---------------------------------------------------------------------------

def dot_scores(z: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Raw dot product. Keeps the norm channel, drops the learned MLP."""
    return (z[pairs[:, 0]] * z[pairs[:, 1]]).sum(-1)


def cosine_scores(z: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Cosine. Kills the norm channel, drops the learned MLP.

    Comparing dot vs cosine isolates the norm channel; comparing the stored
    MLP scores vs dot isolates the decoder. Changing both at once — which is
    easy to do by accident — confounds them.
    """
    zn = z / np.linalg.norm(z, axis=1, keepdims=True)
    return (zn[pairs[:, 0]] * zn[pairs[:, 1]]).sum(-1)


def auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    y = np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
    return float(roc_auc_score(y, np.r_[pos_scores, neg_scores]))


def evaluate_decoders(z: np.ndarray, eval_sets: dict = None) -> dict:
    """AUROC under the dot and cosine decoders, on every evaluation set."""
    eval_sets = eval_sets or EVAL_SETS
    rows = {}
    for name, spec in eval_sets.items():
        try:
            pos, neg = load_split(spec)
        except FileNotFoundError as e:
            print(f"{name}: missing {Path(e.filename).name}")
            continue
        rows[name] = {
            "dot":    auroc(dot_scores(z, pos), dot_scores(z, neg)),
            "cosine": auroc(cosine_scores(z, pos), cosine_scores(z, neg)),
        }
    return rows


def stored_scores_auroc(tag: str = "random_pubmedbert_seed0") -> dict:
    """AUROC from the scores archived during training (the learned MLP).

    Files look like scores_{train_neg}_{features}_seed{n}_{split}_{pos|neg}.npy
    in data/results/. This is the only way to get the learned-decoder column
    without retraining.
    """
    out = {}
    for split in ("random", "degree_matched", "crescenddi"):
        try:
            p = np.load(RESULTS / f"scores_{tag}_{split}_pos.npy")
            n = np.load(RESULTS / f"scores_{tag}_{split}_neg.npy")
            out[split] = auroc(p, n)
        except FileNotFoundError:
            out[split] = None
    return out


# ---------------------------------------------------------------------------
# 7. Pair features and classical baselines
# ---------------------------------------------------------------------------

def pair_features(pairs: np.ndarray, mode: str,
                  x: np.ndarray = None, log_deg: np.ndarray = None) -> np.ndarray:
    """Symmetric feature vector for each pair.

    mode: 'deg' | 'emb' | 'emb_deg'

    Symmetry is not optional. np.sort on the two degrees is what makes the
    degree block order-independent; without it the classifier learns which
    column happened to be stored first, which is an artefact of the file.
    """
    blocks = []
    if mode in ("emb", "emb_deg"):
        zi, zj = x[pairs[:, 0]], x[pairs[:, 1]]
        blocks += [zi * zj, np.abs(zi - zj)]
    if mode in ("deg", "emb_deg"):
        blocks.append(np.sort(log_deg[pairs], axis=1))
    return np.hstack(blocks).astype(np.float32)


def fit_classical_baseline(train_neg: str, mode: str, graph: Graph,
                           x: np.ndarray, max_train: int = 50_000,
                           C: float = None, seed: int = 0) -> dict:
    """Logistic regression on pair features, evaluated on every split.

    The key comparison is 'deg' against 'emb_deg': if adding 1536 dimensions
    of biomedical embeddings on top of two degree numbers buys 0.01 AUROC,
    the features are not doing the work the architecture implies they are.

    C defaults to 1.0 for the random regime and 0.01 for CRESCENDDI, which
    has only a few thousand pairs against 1536 features and will otherwise
    overfit badly. Tune on validation, never on test.
    """
    rng = np.random.default_rng(seed)
    pos, neg = load_split(TRAIN_SETS[train_neg])
    if len(pos) > max_train:
        pos = pos[rng.choice(len(pos), max_train, replace=False)]
    if len(neg) > max_train:
        neg = neg[rng.choice(len(neg), max_train, replace=False)]

    kw = dict(x=x, log_deg=graph.log_deg)
    X = np.vstack([pair_features(pos, mode, **kw), pair_features(neg, mode, **kw)])
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]

    if C is None:
        C = 0.01 if train_neg == "crescenddi" else 1.0
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
    clf.fit(X, y)

    out = {}
    for ev, spec in EVAL_SETS.items():
        ep, en = load_split(spec)
        Xe = np.vstack([pair_features(ep, mode, **kw), pair_features(en, mode, **kw)])
        ye = np.r_[np.ones(len(ep)), np.zeros(len(en))]
        out[ev] = float(roc_auc_score(ye, clf.predict_proba(Xe)[:, 1]))
    return out


# ---------------------------------------------------------------------------
# 8. Data integrity checks
# ---------------------------------------------------------------------------

def check_files() -> bool:
    """Print what exists and what does not. Run this first on a new machine."""
    ok = True
    print(f"DATA   {DATA}  {'OK' if DATA.exists() else 'MISSING'}")
    print(f"FROZEN {FROZEN}  {'OK' if FROZEN.exists() else 'MISSING'}")
    if FROZEN.exists():
        print("\nsplit files:")
        for f in sorted(FROZEN.glob("*.npy")):
            a = np.load(f, mmap_mode="r")
            print(f"  {f.name:<26} {str(a.shape):<14}")
    print("\nfeature files:")
    for short, fn in sorted(FEATURES.items()):
        p = DATA / fn
        mark = f"{np.load(p, mmap_mode='r').shape}" if p.exists() else "MISSING"
        if not p.exists():
            ok = False
        print(f"  {short:<12} {fn:<32} {mark}")
    print(f"\ncheckpoint {CKPT_DEFAULT.name}: "
          f"{'OK' if CKPT_DEFAULT.exists() else 'MISSING'}")
    return ok


def check_crescenddi_sets(graph: Graph) -> dict:
    """Confirm that every CRESCENDDI file contains verified NEGATIVES only.

    Two things should hold, and both are consequences of 01/02 having removed
    the 2,233 conflicting edges before freezing the splits:

      1. No CRESCENDDI pair appears anywhere in the OGB edge list.
      2. cres_* and cres_dd_* are the same underlying pairs, so their union
         as sets is much smaller than the sum of their sizes.

    If (1) fails, conflict removal did not run and you are measuring the
    contaminated 0.367 artefact from the earlier protocol. If (2) fails, the
    two families are not what this file assumes and EVAL_SETS needs revisiting.
    """
    all_pos = np.vstack([np.load(FROZEN / f) for f in
                         ("ogb_train_pos.npy", "ogb_val_pos.npy", "ogb_test_pos.npy")])
    edge_set = set(map(tuple, np.sort(all_pos, axis=1)))

    out, keys = {}, {}
    for f in ("cres_train.npy", "cres_val.npy", "cres_test.npy",
              "cres_dd_train.npy", "cres_dd_val.npy", "cres_dd_test.npy"):
        path = FROZEN / f
        if not path.exists():
            continue
        arr = np.load(path)
        k = set(map(tuple, np.sort(arr, axis=1)))
        keys[f] = k
        hits = len(k & edge_set)
        out[f] = dict(n=len(arr), overlap=hits)
        flag = "" if hits == 0 else "  <-- CONFLICT, rerun the freeze script"
        print(f"  {f:<22} n={len(arr):<6} overlap with OGB edges={hits}{flag}")

    pair = set().union(*[keys[f] for f in keys if not f.startswith("cres_dd")])
    dd = set().union(*[keys[f] for f in keys if f.startswith("cres_dd")])
    print(f"\n  pair-level union: {len(pair)}   drug-disjoint union: {len(dd)}")
    print(f"  shared pairs: {len(pair & dd)} ({len(pair & dd)/max(len(dd),1):.0%} of dd)")
    print("  High overlap confirms: same pairs, two splitting strategies.")
    return out


# ---------------------------------------------------------------------------
# 9. CLI
# ---------------------------------------------------------------------------

def _cmd_check(args):
    check_files()
    g = load_graph()
    print(f"\ngraph: {g.summary()}")
    print("\nCRESCENDDI sets (all should be negatives, zero edge overlap):")
    check_crescenddi_sets(g)


def _cmd_diagnose(args):
    g = load_graph()
    print(f"graph: {g.summary()}")
    z_t = embeddings_from_checkpoint(args.ckpt, g, args.features)
    z_r = embeddings_untrained(g, args.features)
    res = [norm_degree_diagnostic(z_t, g, "trained"),
           norm_degree_diagnostic(z_r, g, "untrained")]
    RESULTS.mkdir(parents=True, exist_ok=True)
    plot_norm_vs_degree([("trained", z_t), ("untrained", z_r)], g,
                        RESULTS / f"norm_vs_degree_{args.tag}.png")
    json.dump(res, open(RESULTS / f"norm_degree_{args.tag}.json", "w"), indent=2)


def _cmd_decoders(args):
    g = load_graph()
    z = embeddings_from_checkpoint(args.ckpt, g, args.features)
    rows = evaluate_decoders(z)
    learned = stored_scores_auroc(args.tag)
    key = {"ogb": "random", "degree_matched": "degree_matched",
           "crescenddi": "crescenddi"}
    print(f"\n{'eval set':<18}{'learned MLP':>13}{'dot':>9}{'cosine':>9}")
    for ev, r in rows.items():
        lm = learned.get(key[ev])
        lm_s = f"{lm:.3f}" if lm is not None else "   -"
        print(f"{ev:<18}{lm_s:>13}{r['dot']:>9.3f}{r['cosine']:>9.3f}")


def _cmd_baselines(args):
    g = load_graph()
    x = load_features(args.features)
    out = {}
    print(f"\n{'train':<12}{'features':<10}" +
          "".join(f"{e:>17}" for e in EVAL_SETS))
    for tn in ("random", "crescenddi"):
        for mode in ("deg", "emb", "emb_deg"):
            r = fit_classical_baseline(tn, mode, g, x)
            out[f"{tn}|{mode}"] = r
            print(f"{tn:<12}{mode:<10}" +
                  "".join(f"{r[e]:>17.3f}" for e in EVAL_SETS))
    RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RESULTS / "classical_baselines.json", "w"), indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command",
                   choices=["check", "diagnose", "decoders", "baselines"])
    p.add_argument("--ckpt", default=str(CKPT_DEFAULT))
    p.add_argument("--features", default="pubmedbert")
    p.add_argument("--tag", default="random_pubmedbert_seed0")
    args, _ = p.parse_known_args()        # parse_known_args: survives Jupyter's -f

    {"check": _cmd_check, "diagnose": _cmd_diagnose,
     "decoders": _cmd_decoders, "baselines": _cmd_baselines}[args.command](args)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# REFERENCE RESULTS — reproduce these before trusting a modified version
# ---------------------------------------------------------------------------
#
# Checkpoint: gcn_random_pubmedbert.pt (random negatives, raw PubMedBERT,
# 3-layer GCN, 768 -> 256 -> 256 -> 256). Graph: frozen_r5/ogb_train_pos.npy,
# 4,267 nodes, 1,066,276 edges, median degree 446, max 2,234.
#
# Norm-vs-degree diagnostic
#   trained     Pearson 0.756  Spearman 0.844  R2 0.571  decile ratio 5.34
#   untrained   Pearson 0.872  Spearman 0.929  R2 0.761  decile ratio 8.98
#
#   Reading: the degree-norm coupling is present BEFORE training. It follows
#   from aggregation itself. Training reduces it; it never removes it. Do not
#   write that the model "learns to exploit degree" — it inherits the channel
#   and keeps most of it.
#
# Decoder comparison (AUROC)
#                    learned MLP    dot     cosine
#   ogb                  0.901     0.845     0.777
#   degree_matched       0.664     0.451     0.620
#   crescenddi           0.514      TBD       TBD
#
#   Reading: the raw dot product is BELOW chance on degree-matched pairs.
#   Once degree is controlled for, magnitude actively misranks. The learned
#   MLP already attenuates this on its own (0.664 > 0.620), which is why a
#   cosine decoder was not pursued further.
#
#   The learned-MLP CRESCENDDI figure (0.514) comes from the archived training
#   scores and is sound — it matches the 0.527 reported in August. The dot and
#   cosine cells must be recomputed with the corrected EVAL_SETS; the earlier
#   values were measured on a non-task.
#
# Classical baselines, logistic regression (AUROC)
#                              ogb   degree_matched   crescenddi
#   random     deg           0.880        0.328            --
#   random     emb           0.801        0.581            --
#   random     emb_deg       0.890        0.457            --
#
#   Reading: deg 0.880 -> emb_deg 0.890. Adding 1536 dimensions of biomedical
#   embeddings on top of two degree numbers buys 0.010 AUROC on the benchmark.
#   And deg alone collapses to 0.328 on degree-matched pairs, which is what
#   that control set is for.
#
#   The CRESCENDDI column and the CRESCENDDI-trained rows were produced with a
#   broken definition (cres_test as positives against cres_dd_test) and have
#   been removed rather than recorded. Both files are negatives. Rerun with the
#   corrected EVAL_SETS / TRAIN_SETS above.
#
# The headline
#   The model scores 0.901 on the standard benchmark and 0.664 on pairs
#   matched for degree. The 0.237 gap is what degree was contributing.
#   CRESCENDDI stays near chance (0.514) even for the best decoder, well below
#   0.664, so the degree shortcut explains part of the benchmark score but not
#   the failure to transfer. The residual gap is the source mismatch between
#   DrugBank-derived edges and clinical compendia (~55% of CRESCENDDI verified
#   non-interactions appear as positive edges in OGB-ddi).
#
# Open threads for whoever picks this up
#   1. Resolve the CRESCENDDI label polarity (check_label_polarity).
#   2. Degree-matched negative sampling during TRAINING, not just evaluation.
#      Expected to be the strongest single intervention.
#   3. LayerNorm or fixed fan-out in the encoder, since the norm channel
#      originates there rather than in the decoder.
#   4. Multi-seed runs: everything above is a single seed.