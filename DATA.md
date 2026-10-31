# Data reference — DDI link prediction

- Data: `/home/renku/work/data/`
- Splits: `/home/renku/work/data/frozen_r5/` ← **use this one**, not `frozen/`
- Results: `/home/renku/work/data/results/`

## The files in this project

| File | What it is |
|---|---|
| `00_HANDOVER.ipynb` | Guided tour. Start here — runs in minutes, writes nothing. |
| `DATA.md` | This file. Every array on disk and what it means. |
| `ddi_toolkit.py` | Analysis functions: load, rescore, diagnose. No training. |
| `01_freeze_splits.py` | How the splits were built. **Do not run** — it would invalidate stored results. |
| `05_gcn_experiment.py` | Training + evaluation, as a script. |
| `06_train_gcn.ipynb` | Training + evaluation, as a notebook. ~20 min on Renku CPU. |

```bash
python ddi_toolkit.py check        # verify every file is present and sane
python ddi_toolkit.py diagnose     # norm vs degree, with untrained control
python ddi_toolkit.py decoders     # stored scores vs dot vs cosine
python ddi_toolkit.py baselines    # logistic regression on pair features
```

In a notebook, import instead of running: `from ddi_toolkit import *`.

---

## Context in four sentences

OGB-ddi (4,267 drugs, 1.33M edges, density 14.7%) comes from DrugBank, which
includes mechanistically *predicted* interactions. CRESCENDDI (Kontsioti et
al. 2022) provides clinically **verified non-interactions** — the negative
control the benchmark lacks. The two disagree: 55% of CRESCENDDI negatives
appear as OGB edges, mostly a degree effect (degree-matched random pairs hit
59.8%) rather than pure compendium disagreement.

---

## Model configuration — four settings that look wrong and are not

Each of these was recovered by checking against the stored checkpoint and the
per-epoch logs. Changing any of them makes new runs incomparable with every
stored result, so treat a change as a **new experiment**, not a bug fix.

**The decoder is a plain dot product**, `(z_i * z_j).sum(-1)`, with no
parameters. This is the DDI-LLM architecture. Evidence:
`gcn_random_pubmedbert.pt` holds only `convs.0-2` and no decoder keys — a dot
product has none to save. An MLP decoder is also ~256x more expensive on 2.1M
supervision pairs, which turns a 20-minute CPU run into hours.

**`SYMMETRISE = False`.** `ogb_train_pos.npy` stores each undirected edge once
and `GCNConv` does not add the reverse, so messages flow one way only. Unusual
for an undirected graph, but it is what the runs did: reloading the checkpoint
gives mean embedding norm 4.327 and r(norm, log-degree) = 0.756 with `False`,
versus 0.793 and 0.541 with `True`. The first pair matches the reference.

**`STANDARDISE = True`** — a per-column z-score, `(x - x.mean(0)) / x.std(0)`,
**not** an L2 row normalisation. Centring removes the shared mean vector, which
is what makes raw PubMedBERT anisotropic (mean pairwise cosine 0.921). Without
it every drug points roughly the same way and the dot-product logits blow up.
It also puts every feature type on one scale, so the embedding comparison
measures content rather than magnitude.

**The last epoch is reported**, not a validation-selected one. The validation
set uses random negatives, so it is subject to the same degree shortcut as the
benchmark and tells you little; the control sets keep improving until ~300.
Early stopping on validation would cut exactly where the interesting part
starts.

---

## Conflict removal — where it happened

A pair cannot be a positive edge and a verified non-interaction at the same
time. The supervisor's decision was to give CRESCENDDI priority, so the
conflicting pairs were **deleted from the OGB edge list**, not from CRESCENDDI.

**What was removed:** 2,233 edges out of 1,334,889 (0.17%) — every OGB edge
that also appears as a CRESCENDDI pair.

**Where:** in the freeze script, before splitting. The removal applies to all
three positive files, not just the training graph:

```
ogb_train_pos  1,066,276
ogb_val_pos      133,263
ogb_test_pos     133,117
               ---------
total          1,332,656  = 1,334,889 − 2,233
```

**Where it landed:** `frozen_r5/`. The older `frozen/` directory was written
before the removal and still contains the conflicts.

**How to verify:**

```python
from ddi_toolkit import load_graph, check_crescenddi_sets
check_crescenddi_sets(load_graph())    # every overlap count must be 0
```

**If you skip this:** the model is penalised for correctly scoring pairs that
DrugBank calls interactions, and CRESCENDDI AUROC collapses to 0.367. That
number appears in early drafts and is an artefact, not a result.

---

## Files

All arrays are `(n, 2)` int64, OGB node indices in `[0, 4266]`. Pairs are
**unordered** — symmetrise anything you compute from them (sort, or use the
elementwise product).

### Positives (OGB edges)

Conflict-free: no CRESCENDDI pair appears in any of these.

| File | n | Use |
|---|---|---|
| `ogb_train_pos.npy` | 1,066,276 | The message-passing graph. Compute degrees from this only. |
| `ogb_val_pos.npy` | 133,263 | Validation. |
| `ogb_test_pos.npy` | 133,117 | **Positives in every evaluation** — all three rows of the protocol table use these same pairs; only the negatives change. |

### Negatives

| File | n | Use |
|---|---|---|
| `random_neg_train.npy` | 1,066,276 | Training, 1:1 with positives. |
| `random_neg_val.npy` | 20,000 | Validation. |
| `random_neg_test.npy` | 133,117 | Standard benchmark evaluation. |
| `degmatched_neg_test.npy` | 810 | **The control.** Degree distribution matched to `cres_test`. |
| `ogb_val_neg.npy` / `ogb_test_neg.npy` | 101,882 / 95,599 | OGB's own negatives. Unused here. |

### CRESCENDDI — **all negatives, no positives**

Pair-level split (70/10/20): `cres_train` 2,834 · `cres_val` 404 ·
`cres_test` 811. Union = 4,049, the complete set.

Drug-disjoint split of the **same pairs**: `cres_dd_train` 1,853 ·
`cres_dd_val` 39 · `cres_dd_test` 172. Union = 2,064, all contained in the
pair-level union. `dd` = drug-disjoint; the 1,985 missing pairs had drugs on
both sides of a boundary.

### Features — `(4267, d)`, row = node index

| Name | File | Dim | Note |
|---|---|---|---|
| `pubmedbert` | `pubmedbert_embeddings.npy` | 768 | Descriptions. 100% coverage. |
| `chemberta` | `chemberta_embeddings.npy` | 768 | SMILES. 83%; mean-vector fallback. |
| `gpt` | `gpt_embeddings.npy` | 1536 | ada-002. **Only 3,481 rows.** |
| `random1536` | `random1536_embeddings.npy` | 1536 | Gaussian control. |
| `pmbwhite` / `cbwhite` / `bothwhite` | `*_embeddings.npy` | 768/768/1536 | ZCA-whitened variants. |

Whitening exists because raw embeddings are anisotropic (mean pairwise cosine
0.921 for PubMedBERT); ZCA drops it to ~0.02 and reverses the feature ranking
on control sets. This is a stronger transform than the per-column z-score
above, and the two are independent — the z-score is applied to whatever feature
file you load, whitened or not.

### Results — the per-epoch CSVs are the primary record

```
gcn_{train_neg}_{features}_seed{n}.csv
```

7 feature types x 2 regimes x 5 seeds. **These are the authoritative numbers**,
and the first place to look for any figure quoted in the report. Columns:

`run, seed, epoch, features, train_neg, loss, split, neg_set, auroc,
aupr_neg, mean_pos, mean_neg, n_pos, n_neg`

Logged at epoch 1 then every 5, with one row per evaluation set plus `val` and
`train`. So `gcn_random_pubmedbert_seed0.csv` has 405 rows over 400 epochs.

```python
h = pd.read_csv(RESULTS / "gcn_random_pubmedbert_seed0.csv")
h[(h.epoch == 400) & (h.split == "test")][["neg_set", "auroc", "mean_pos", "mean_neg"]]
```

Also present:

- `gcn_random_pubmedbert.pt` — bare `OrderedDict`, **encoder only**. Three
  GCNConv layers, 768→256→256→256. Load with `weights_only=False`. There are no
  decoder weights because the decoder has no parameters. Note that this
  checkpoint does **not** correspond exactly to the epoch-400 CSV row:
  rescoring it with a dot product gives 0.845 where the CSV says 0.901, so it
  is probably a different epoch. Prefer the CSV.
- `scores_{...}_{split}_{pos|neg}.npy` — raw per-pair scores, for recomputing
  any metric without retraining.

---

## Protocol

Same positives, three negative sets. The contrast is the result; no single
number means much alone.

| Negatives | Isolates |
|---|---|
| `random_neg_test` | The benchmark. |
| `degmatched_neg_test` | Removes the degree confound. Gap above = degree shortcut. |
| `cres_test` | Adds clinical verification. Gap above = source mismatch. |

Imbalance is handled by weighting the loss, never by resampling (upsampling was
tried and hurt). Focal loss was also tried; its optimal gamma collapsed to ~0,
reducing it to weighted BCE. With 811 CRESCENDDI negatives the AUROC standard
error is ~0.02 — quote it.

---

## Results

`gcn_random_pubmedbert_seed0`, epoch 400:

| Evaluation | AUROC | mean score, positives | negatives | gap |
|---|---|---|---|---|
| Benchmark | 0.901 | 2.48 | −0.37 | **2.84** |
| Degree-matched | 0.664 | 2.88 | 1.11 | **1.78** |
| CRESCENDDI | 0.514 | 2.54 | 1.84 | **0.70** |

The score columns show the mechanism better than the AUROCs. Positives score
the same everywhere, around 2.5. What changes is the **negatives**: −0.37, then
1.11, then 1.84. The model does not get worse at recognising real interactions
— it loses the ability to reject non-interactions as they take on the degree
profile of real edges.

**At epoch 1**, after a single gradient step, the GCN is already numerically
indistinguishable from a three-parameter logistic regression on degree:

| | GCN, epoch 1 | logistic regression on log-degree |
|---|---|---|
| Benchmark | 0.863 | 0.880 |
| Degree-matched | 0.306 | 0.328 |
| CRESCENDDI | 0.308 | 0.329 |

Both land **below chance** on the controls, because CRESCENDDI pairs have a
*higher* mean degree (966.9) than real edges (793.8) — the degree signal does
not merely stop working there, it inverts. Two independent methods converging
on the same three numbers is the strongest form of this finding.

The remaining 399 epochs buy 0.038 on the benchmark and 0.358 on degree-matched
pairs. Adding 1,536 embedding dimensions on top of the two degree features buys
0.010.

Embedding norms correlate with degree at r=0.756 — and at **0.872 before
training**. The coupling comes from neighbourhood aggregation, not from
learning. Write "inherits the channel", not "learns to exploit degree".

### Reproducibility status

`06_train_gcn.ipynb` reproduces the **structure**: same ordering, same large
spread, same conclusion, final loss 0.425 against 0.432, final validation
0.9238 against 0.9219. It does not reproduce the figures exactly — a fresh run
gives 0.915 / 0.724 / 0.550 against 0.901 / 0.664 / 0.514. The benchmark figure
sits within seed noise (0.902 ± 0.010 over 5 seeds); the controls, with 810
pairs each, do not.

The residual gap traces to initialisation scale: epoch-1 loss is 46.7 in a
fresh run against 5.5 in the logs, so some scaling detail of the original run
is still unaccounted for. It changes no conclusion — both sets of numbers show
the same collapse — but quote the CSV, not a re-run, for any figure in the
report.

---

## Traps

1. `frozen/` is pre-conflict-removal. Using it gives a spurious 0.367.
2. All CRESCENDDI files are negatives. `cres_test` vs `cres_dd_test` is a non-task.
3. Feature file must match the checkpoint name. Nothing checks this.
4. `gpt_embeddings.npy` has 3,481 rows, not 4,267 — it will not index.
5. `SYMMETRISE = False` and `STANDARDISE = True` are deliberate. See above.
6. The dot-product decoder is deliberate. An MLP is a different, much slower model.
7. `torch.load` needs `weights_only=False` (PyTorch ≥ 2.6).
8. argparse in Jupyter: `parse_known_args()`, defaults, no `required=True`.
9. Renku has no usable GPU (driver too old, torch falls back to CPU). With the
   dot-product decoder that is fine — 20 minutes for 400 epochs.

---

## Next steps

1. **Degree-matched negatives during training**, not only at evaluation. Draw
   each negative with the degree profile of the positive it replaces, so degree
   becomes non-discriminative and gradient descent has no shortcut to take.
   This addresses the cause rather than the symptom.
2. **Symmetrise the edge index** as a controlled experiment. It alone drops
   r(norm, log-degree) from 0.756 to 0.541 — the degree channel weakening — so
   it may be a more direct intervention than anything at the decoder. A cosine
   decoder was tested post-hoc and reached only 0.620 on degree-matched pairs,
   below the 0.664 of the dot product, so that route was dropped.
3. **`LayerNorm`, fixed fan-out, or DropEdge in the encoder** — the norm
   channel originates in aggregation, which is where to intervene.
4. **One-hot ATC vectors** as an ablation against semantic embeddings, to
   separate broad therapeutic structure from fine-grained pharmacology.
5. **Close the reproducibility gap**: find the initialisation detail that makes
   epoch-1 loss 5.5 rather than 46.7.

---

## References

Hu et al. 2020 (OGB) · Kontsioti et al. 2022 (CRESCENDDI — *not* Vilar et al.)
· Kipf & Welling 2017 (GCN) · Guney 2017 (benchmark artefacts) · Giobbi 2026
(BioSnap precedent; the density contrast, 4.2% against 14.7%, is why that work
did not hit the overlap problem).

Supervisors: Paraskevi Nousi (Vivi), Maxim Samarin — SDSC.
