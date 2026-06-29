# Results

This file records the **verified, contamination-controlled** results. Earlier inflated numbers
(e.g. a single-dataset AUC of ~0.82) came from a contaminated preprocessing path and are
**refuted** — they must not be used.

## 1. Synthetic validation (theory)

| Test | Result |
|---|---|
| K=2 closed-form parameter recovery (zero noise) | relative error <= 1e-15 |
| K=5 prerequisite-graph recovery | F1 = 1.0 |
| K=5 predictive AUC | 0.717 = oracle ceiling (0.720) |

These confirm the identifiability theorem numerically under the required excitation schedule.

## 2. Clean real-data accuracy (the honest trade-off)

Matched student-level folds; identical keys across models; DeLong significance.

| Dataset | CPINN | DKT | BKT |
|---|---|---|---|
| ASSISTments-2009 | 0.682 | 0.758 | 0.717 |
| ASSISTments-2015 | 0.695 | 0.730 | 0.691 |
| ASSISTments-2017 | 0.607 | 0.697 | 0.627 |

All pairwise differences are statistically significant (DeLong, p ~ 0).
**Cognitive-PINN is the weakest model on accuracy** — below DKT everywhere and below BKT on two
of three datasets. This is a deliberate trade-off: the structural constraints that buy
identifiability and interpretability reduce predictive flexibility. Accuracy is **not** a
contribution of this work.

## 3. External validation on Junyi (the main empirical result)

Cognitive-PINN was trained on a 25k-student subsample of Junyi-2015 (K = 40 topics) **without
any expert prior** (Q_prior = 1), then its recovered parameters were compared against external
references.

### 3.1 Parameters: forgetting Lambda vs topic difficulty — strong

| Test | rho (Spearman) | p |
|---|---|---|
| Lambda vs topic difficulty (error-rate) | 0.917 | 2e-16 |
| Lambda vs difficulty, controlling for observation count (partial) | 0.857 | 3e-12 |

The recovered forgetting rates almost perfectly track independently measured topic difficulty,
and the relationship survives controlling for data volume. The model recovered a meaningful,
measurable quantity from response dynamics alone.

![Lambda vs difficulty](figures/fig_lambda_difficulty.png)

### 3.2 Structure: transfer matrix A vs expert prerequisite graph — weak but significant

| Reference graph | edge-AUC | p (null vs random graphs) |
|---|---|---|
| Expert graph e1 (52 edges, primary) | 0.581 | 0.025 (significant) |
| Expert graph e2 (14 edges, sparse) | 0.468 | 0.65 (not significant) |

The recovered structure ranks true prerequisite edges above chance on the primary graph
(significant), but does not reproduce the graph in full, and the signal is not robust on the
sparse threshold. This is consistent with the identifiability theorem: structural recovery
requires excitation conditions that observational logs do not contain (on synthetic data with
the required excitation, structure recovery is perfect, F1 = 1.0). The negative result thus
characterises the **boundary of applicability** of the method.

![Structure significance](figures/fig_structure_significance.png)

## 4. Engineering

| Item | Result |
|---|---|
| Training speedup (architecture iterations) | ~400x |
| Batched solver vs per-student path | predictions bit-exact; gradients ~1e-10; ~245x faster on large data |

## Reproducibility notes

- Clean real-data and Junyi runs use `lambda_ode = 0` (per-student and batched objectives identical).
- The structure experiment additionally disables the L1/L2 penalty on B (ablation), since the L1
  penalty drives the off-diagonal of A toward zero and suppresses the structure being tested.
- ASSISTments-2009 duplicated rows (Xiong et al., 2016) were removed before all runs.
