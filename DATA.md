# Data reference — DDI link prediction

Companion to `ddi_toolkit.py`, which loads all of this and reproduces the
numbers at the bottom.

- Data: `/home/renku/work/data/`
- Splits: `/home/renku/work/data/frozen_r5/` ← **use this one**, not `frozen/`
- Results: `/home/renku/work/data/results/`

---

## Quick start

```bash
python ddi_toolkit.py check        # verify every file is present and sane
python ddi_toolkit.py diagnose     # norm vs degree, with untrained control
python ddi_toolkit.py decoders     # learned MLP / dot / cosine comparison
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
59.8%) rather than pure compendium disagreement. The 2,233 conflicting edges
were removed; `frozen_r5/` is the result.

---

## Files

All arrays are `(n, 2)` int64, OGB node indices in `[0, 4266]`. Pairs are
**unordered** — symmetrise anything you compute from them.

### Positives (OGB edges)

| File | n | Use |
|---|---|---|
| `ogb_train_pos.npy` | 1,066,276 | The message-passing graph. Compute degrees from this only. |
| `ogb_val_pos.npy` | 133,263 | Validation. |
| `ogb_test_pos.npy` | 133,117 | **Positives in every evaluation.** |

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
0.921 for PubMedBERT); whitening drops it to ~0.02 and reverses the feature
ranking on control sets.

### Results

`gcn_random_pubmedbert.pt` — bare `OrderedDict`, **encoder only**. Three
GCNConv layers, 768→256→256→256, ReLU between but not after. Load with
`weights_only=False`.

`scores_{train_neg}_{features}_seed{n}_{split}_{pos|neg}.npy` — the learned
decoder's outputs. The only way to recover the trained-decoder column without
retraining, since decoder weights were never saved.

---

## Protocol

Same positives, three negative sets. The contrast is the result.

| Negatives | Isolates |
|---|---|
| `random_neg_test` | The benchmark. |
| `degmatched_neg_test` | Removes the degree confound. Gap above = degree shortcut. |
| `cres_test` | Adds clinical verification. Gap above = source mismatch. |

Imbalance is handled by weighting the loss, never by resampling (upsampling
was tried and hurt). With 811 CRESCENDDI negatives, AUROC standard error is
~0.02 — quote it.

---

## Results

| Evaluation | AUROC |
|---|---|
| Benchmark | 0.901 |
| Degree-matched | 0.664 |
| CRESCENDDI | 0.514 |

The 0.237 gap is what degree contributed. Logistic regression on two
log-degrees alone reaches 0.880; adding 1,536 embedding dimensions buys 0.010.

Embedding norms correlate with degree at r=0.756 — and at **0.872 before
training**. The coupling comes from neighbourhood aggregation, not from
learning. Write "inherits the channel", not "learns to exploit degree".

---

## Traps

1. `frozen/` is pre-conflict-removal. Using it gives a spurious 0.367.
2. All CRESCENDDI files are negatives. `cres_test` vs `cres_dd_test` is a non-task.
3. Feature file must match the checkpoint name. Nothing checks this.
4. `gpt_embeddings.npy` has 3,481 rows, not 4,267.
5. `GCNConv` does not symmetrise — flip and concatenate the edge index.
6. `torch.load` needs `weights_only=False` (PyTorch ≥ 2.6).
7. argparse in Jupyter: `parse_known_args()`, defaults, no `required=True`.

---

## Next steps

1. Degree-matched negatives during **training**, not just evaluation. Likely
   the strongest single intervention.
2. `LayerNorm` or DropEdge in the encoder — the norm channel originates there.
3. One-hot ATC vectors as an ablation against semantic embeddings.
4. Multi-seed runs for the decoder and diagnostic results (currently n=1).

---

## References

Hu et al. 2020 (OGB) · Kontsioti et al. 2022 (CRESCENDDI — *not* Vilar et
al.) · Kipf & Welling 2017 (GCN) · Guney 2017 (benchmark artefacts) · Giobbi
2026 (BioSnap precedent).

Supervisors: Paraskevi Nousi (Vivi), Maxim Samarin — SDSC.
