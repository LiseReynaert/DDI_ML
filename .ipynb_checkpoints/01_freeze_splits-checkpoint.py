"""
01_freeze_splits.py — Build every dataset split, once, and freeze it to disk.

WARNING — DO NOT RUN THIS unless you intend to invalidate every stored result.
The splits in data/frozen_r5/ were produced by this script with seed 0. Every
checkpoint and every scores_*.npy file in data/results/ was computed against
them. Re-running with a different seed, or after any change below, silently
breaks the comparability of everything already measured.

The original ran in April 2026; this is a reconstruction of that script from its outputs and from the working notes.
Verify against the existing arrays before trusting a regenerated set:

    python 01_freeze_splits.py --out /tmp/frozen_check --seed 0
    python -c "
    import numpy as np
    for f in ['cres_test','degmatched_neg_test','random_neg_test']:
        a = np.load('/home/renku/work/data/frozen_r5/%s.npy' % f)
        b = np.load('/tmp/frozen_check/%s.npy' % f)
        print(f, a.shape == b.shape, np.array_equal(np.sort(a,1), np.sort(b,1)))
    "

What it produces
----------------
  ogb_{train,val,test}_pos     OGB edges, conflicting pairs removed
  ogb_{val,test}_neg           OGB's own negatives (passed through unchanged)
  cres_{train,val,test}        CRESCENDDI verified negatives, pair-level split
  cres_dd_{train,val,test}     the same pairs, drug-disjoint split
  random_neg_{train,val,test}  uniform non-edges
  degmatched_neg_test          non-edges matched to cres_test's degree profile

The central design decision
---------------------------
A pair cannot be both a DrugBank interaction and a clinically verified
non-interaction. 2,233 OGB edges (0.17% of 1,334,889) are also CRESCENDDI
pairs. The supervisor's call was to give CRESCENDDI priority: those edges are
deleted from the OGB edge list, across all three splits. CRESCENDDI then
becomes an evaluation set rather than a source of training negatives.

Without this step the model is penalised for correctly scoring pairs DrugBank
calls interactions, and CRESCENDDI AUROC reads 0.367 — an artefact that
appears in early drafts.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/home/renku/work/data")
N_NODES = 4267

# CRESCENDDI split fractions; test takes the remainder.
FRAC_TRAIN, FRAC_VAL = 0.70, 0.20

# Random negatives: 1:1 with positives on train and test, smaller on val.
N_RANDOM_VAL = 20_000


# ---------------------------------------------------------------------------
# Pair utilities — everything here must be order-independent
# ---------------------------------------------------------------------------

def pair_keys(pairs: np.ndarray) -> np.ndarray:
    """One int64 key per unordered pair, for set membership.

    Sorting first is what makes (i,j) and (j,i) collide, which is required:
    nothing guarantees the column order in any source file.
    """
    p = np.sort(pairs, axis=1)
    return p[:, 0].astype(np.int64) * N_NODES + p[:, 1].astype(np.int64)


def compute_degrees(edges: np.ndarray) -> np.ndarray:
    """Node degree from an edge list. Pass TRAINING edges only."""
    deg = np.zeros(N_NODES, dtype=np.int64)
    np.add.at(deg, edges[:, 0], 1)
    np.add.at(deg, edges[:, 1], 1)
    return deg


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_pairwise(pairs: np.ndarray, rng) -> tuple:
    """Shuffle and cut 70/20/10. Drugs may appear in more than one split."""
    idx = rng.permutation(len(pairs))
    n_tr = int(FRAC_TRAIN * len(pairs))
    n_va = int(FRAC_VAL * len(pairs))
    return (pairs[idx[:n_tr]],
            pairs[idx[n_tr:n_tr + n_va]],
            pairs[idx[n_tr + n_va:]])


def split_drug_disjoint(pairs: np.ndarray, rng) -> tuple:
    """Assign DRUGS to splits, then keep only pairs whose both drugs agree.

    Harder evaluation: a test pair involves no drug seen in training. The cost
    is high — on CRESCENDDI this drops 1,985 of 4,049 pairs (49%), leaving
    2,064. Cross-boundary pairs cannot be assigned anywhere without leaking.
    """
    drugs = np.unique(pairs)
    perm = rng.permutation(len(drugs))
    n_tr = int(FRAC_TRAIN * len(drugs))
    n_va = int(FRAC_VAL * len(drugs))

    assign = {}
    for k, grp in enumerate([perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]]):
        for d in drugs[grp]:
            assign[int(d)] = k

    buckets = [[], [], []]
    for a, b in pairs:
        ka, kb = assign[int(a)], assign[int(b)]
        if ka == kb:
            buckets[ka].append((a, b))
    return tuple(np.array(b, dtype=np.int64).reshape(-1, 2) for b in buckets)


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def sample_random_negatives(n: int, forbidden: set, rng,
                            max_tries: int = 200) -> np.ndarray:
    """Uniform non-edges, excluding everything in `forbidden`.

    `forbidden` must contain all graph edges AND all CRESCENDDI pairs, so that
    a random negative can never collide with a verified one. It is also grown
    between calls so the three splits stay disjoint.
    """
    out, seen = [], set()
    for _ in range(max_tries):
        need = n - len(out)
        if need <= 0:
            break
        cand = rng.integers(0, N_NODES, size=(int(need * 1.5) + 100, 2))
        cand = cand[cand[:, 0] != cand[:, 1]]
        keys = pair_keys(cand)
        for p, k in zip(cand, keys):
            if k in forbidden or k in seen:
                continue
            seen.add(int(k))
            out.append(p)
            if len(out) == n:
                break
    if len(out) < n:
        raise RuntimeError(f"only sampled {len(out)}/{n} negatives")
    return np.array(out, dtype=np.int64)


def sample_degree_matched(target: np.ndarray, deg: np.ndarray,
                          forbidden: set, rng, n_bins: int = 20,
                          max_tries: int = 500) -> np.ndarray:
    """For each target pair, a non-edge whose two endpoints have similar degree.

    This is the control set the whole project turns on. On these pairs degree
    carries no discriminative information by construction, so whatever AUROC
    survives is not the degree shortcut.

    Matching is done in log-degree quantile bins rather than on exact degree:
    exact matching fails too often in a graph this dense.
    """
    log_deg = np.log1p(deg)
    edges = np.quantile(log_deg, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.digitize(log_deg, edges[1:-1]), 0, n_bins - 1)
    by_bin = {b: np.where(bins == b)[0] for b in range(n_bins)}

    out, seen = [], set()
    for a, b in target:
        ba, bb = bins[a], bins[b]
        for _ in range(max_tries):
            i = int(rng.choice(by_bin[ba]))
            j = int(rng.choice(by_bin[bb]))
            if i == j:
                continue
            k = int(pair_keys(np.array([[i, j]]))[0])
            if k in forbidden or k in seen:
                continue
            seen.add(k)
            out.append((i, j))
            break
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def report_match_quality(target: np.ndarray, matched: np.ndarray,
                         deg: np.ndarray):
    """Did the matching work? Mean endpoint degree should be close."""
    dt = np.concatenate([deg[target[:, 0]], deg[target[:, 1]]])
    dm = np.concatenate([deg[matched[:, 0]], deg[matched[:, 1]]])
    print(f"  target  mean={dt.mean():8.1f}  median={np.median(dt):8.1f}")
    print(f"  matched mean={dm.mean():8.1f}  median={np.median(dm):8.1f}")
    print(f"  ratio of means: {dm.mean()/dt.mean():.3f}   (want ~1.00)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(DATA / "frozen_r5"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--crescenddi",
                   default=str(DATA / "crescenddi_ogbddi_usable_negatives.csv"))
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing output directory")
    args, _ = p.parse_known_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(
            f"{out} already exists. Re-running invalidates every stored result.\n"
            f"Pass --force if you are certain, or --out /tmp/... to compare.")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # --- [1] OGB -----------------------------------------------------------
    print("[1] loading OGB-ddi")
    from ogb.linkproppred import PygLinkPropPredDataset
    ds = PygLinkPropPredDataset(name="ogbl-ddi", root=str(DATA))
    sp = ds.get_edge_split()
    ogb = {
        "train_pos": sp["train"]["edge"].numpy(),
        "val_pos":   sp["valid"]["edge"].numpy(),
        "test_pos":  sp["test"]["edge"].numpy(),
        "val_neg":   sp["valid"]["edge_neg"].numpy(),
        "test_neg":  sp["test"]["edge_neg"].numpy(),
    }
    n_before = sum(len(ogb[k]) for k in ("train_pos", "val_pos", "test_pos"))
    print(f"  positives before removal: {n_before:,}")

    # --- [2] CRESCENDDI and conflict removal -------------------------------
    print("\n[2] CRESCENDDI, and removing conflicting edges")
    cres = pd.read_csv(args.crescenddi)[["Drug1", "Drug2"]].to_numpy().astype(np.int64)
    cres_keys = set(pair_keys(cres).tolist())
    print(f"  CRESCENDDI pairs: {len(cres):,}")

    removed = 0
    for k in ("train_pos", "val_pos", "test_pos"):
        keys = pair_keys(ogb[k])
        keep = np.array([kk not in cres_keys for kk in keys])
        removed += int((~keep).sum())
        ogb[k] = ogb[k][keep]
    n_after = sum(len(ogb[k]) for k in ("train_pos", "val_pos", "test_pos"))
    print(f"  removed {removed:,} edges ({removed/n_before:.3%})")
    print(f"  positives after removal: {n_after:,}")

    all_edges = np.concatenate([ogb["train_pos"], ogb["val_pos"], ogb["test_pos"]])
    edge_keys = set(pair_keys(all_edges).tolist())
    assert not (edge_keys & cres_keys), "conflicts remain after removal"
    print("  verified: zero remaining edge/CRESCENDDI overlap")

    # --- [3] CRESCENDDI splits ---------------------------------------------
    print("\n[3] splitting CRESCENDDI (all of these are NEGATIVES)")
    c_tr, c_va, c_te = split_pairwise(cres, rng)
    print(f"  pair-level    train={len(c_tr):,}  val={len(c_va):,}  test={len(c_te):,}")
    d_tr, d_va, d_te = split_drug_disjoint(cres, rng)
    dropped = len(cres) - len(d_tr) - len(d_va) - len(d_te)
    print(f"  drug-disjoint train={len(d_tr):,}  val={len(d_va):,}  test={len(d_te):,}"
          f"   (dropped {dropped:,} cross-split)")

    # --- [4] random negatives ----------------------------------------------
    print("\n[4] sampling random negatives (excluding edges + all CRESCENDDI)")
    forbidden = edge_keys | cres_keys
    rand_te = sample_random_negatives(len(ogb["test_pos"]), forbidden, rng)
    forbidden |= set(pair_keys(rand_te).tolist())
    rand_va = sample_random_negatives(N_RANDOM_VAL, forbidden, rng)
    forbidden |= set(pair_keys(rand_va).tolist())
    rand_tr = sample_random_negatives(len(ogb["train_pos"]), forbidden, rng)
    print(f"  train={len(rand_tr):,}  val={len(rand_va):,}  test={len(rand_te):,}")

    # --- [5] degree-matched control ----------------------------------------
    print("\n[5] degree-matched control for the CRESCENDDI test pairs")
    deg = compute_degrees(ogb["train_pos"])
    dm_forbidden = edge_keys | cres_keys | set(pair_keys(rand_te).tolist())
    dm_te = sample_degree_matched(c_te, deg, dm_forbidden, rng)
    print(f"  matched {len(dm_te):,} of {len(c_te):,}")
    report_match_quality(c_te, dm_te, deg)

    # --- [6] save -----------------------------------------------------------
    print("\n[6] saving")
    to_save = {
        "ogb_train_pos": ogb["train_pos"], "ogb_val_pos": ogb["val_pos"],
        "ogb_test_pos": ogb["test_pos"],
        "ogb_val_neg": ogb["val_neg"], "ogb_test_neg": ogb["test_neg"],
        "cres_train": c_tr, "cres_val": c_va, "cres_test": c_te,
        "cres_dd_train": d_tr, "cres_dd_val": d_va, "cres_dd_test": d_te,
        "random_neg_train": rand_tr, "random_neg_val": rand_va,
        "random_neg_test": rand_te,
        "degmatched_neg_test": dm_te,
    }
    for name, arr in to_save.items():
        np.save(out / f"{name}.npy", arr)
        print(f"  {name:<22} {arr.shape}")

    json.dump({"seed": args.seed, "n_nodes": N_NODES,
               "edges_removed": removed, "positives_after": n_after,
               "decision": "conflicts removed from OGB; CRESCENDDI given priority",
               "sizes": {k: list(v.shape) for k, v in to_save.items()}},
              open(out / "manifest.json", "w"), indent=2)

    # --- [7] audit ----------------------------------------------------------
    print("\n" + "=" * 68)
    print("AUDIT: mean endpoint degree by set  (put this on a slide)")
    print("=" * 68)
    print(f"{'set':<26}{'n':>10}{'mean deg':>12}{'median deg':>12}")
    for name, arr in [("positives (test)", ogb["test_pos"]),
                      ("random negatives", rand_te),
                      ("degree-matched", dm_te),
                      ("CRESCENDDI (test)", c_te)]:
        d = np.concatenate([deg[arr[:, 0]], deg[arr[:, 1]]])
        print(f"{name:<26}{len(arr):>10,}{d.mean():>12.1f}{np.median(d):>12.1f}")
    print("\nCRESCENDDI pairs sit at a HIGHER mean degree than real edges. That "
          "is the\nconfound the degree-matched set exists to remove.")
    print("\nDone.")


if __name__ == "__main__":
    main()