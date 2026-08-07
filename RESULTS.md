# Results

This file records the **verified** results after a full battery of controls. Earlier claims that
did not survive those controls are listed as **refuted** at the bottom; they must not be used.

## 1. Synthetic validation (theory holds under excitation)

| Test | Result |
|---|---|
| K=2 closed-form parameter recovery (zero noise) | relative error <= 1e-15 |
| K=5 prerequisite-graph recovery | F1 = 1.0 |
| K=5 predictive AUC | 0.717 = oracle ceiling (0.720) |

Under a practice schedule satisfying the excitation conditions of Theorem 1, parameters and
coupling structure are recovered exactly. This confirms the theory numerically.

## 2. Clean real-data accuracy (documented trade-off)

Matched student-level folds; identical keys across models; DeLong significance.

| Dataset | CPINN | DKT | BKT |
|---|---|---|---|
| ASSISTments-2009 | 0.682 | 0.758 | 0.717 |
| ASSISTments-2015 | 0.695 | 0.730 | 0.691 |
| ASSISTments-2017 | 0.607 | 0.697 | 0.627 |

All pairwise differences are statistically significant (DeLong, p ~ 0). Cognitive-PINN is the
weakest predictor — a deliberate cost of the structural constraints that buy identifiability.
Accuracy is **not** a contribution of this work.

## 3. Observational data (Junyi): identifiability degenerates

Cognitive-PINN was trained on a 25k-student subsample of Junyi-2015 (K = 39 topics) **without any
expert prior** (Q_prior = 1), and the recovered parameters were tested against external references.
Unless stated otherwise: fold 0, batched solver, `lambda_ode = 0`, `lambda_l1 = lambda_l2 = 0`,
25 fixed epochs (no early stopping).

### 3.1 The apparent result and why it does not hold

Recovered forgetting rates Lambda correlate strongly with independently measured topic difficulty
(error rate): Spearman rho = 0.83 +- 0.01 across five seeds (partial rho | observation count =
0.78 +- 0.01). Taken alone this looks like successful parameter recovery. **Four controls show it
is not evidence of recovering forgetting.**

| # | Control | Result | Reading |
|---|---|---|---|
| C1 | Shuffle interaction order within student (destroys temporal structure, preserves per-topic frequencies) | rho = 0.829 (vs 0.845 unshuffled) | correlation survives -> Lambda tracks error frequency, not dynamics |
| C2 | Same correlation for a standard BKT baseline | slip: rho = 0.795; guess: rho = -0.699 | a 1994 model reproduces it -> not a property of our model |
| C3 | Retrain on **real elapsed time** (log-compressed, quantised) and correlate Lambda with empirical per-topic forgetting slope | rho = 0.49 +- 0.01, while rho vs difficulty stays 0.79 | Lambda still tracks difficulty first |
| C4 | Same, with real time switched off (interaction index) | rho = 0.504 — **not lower** | real timestamps give no gain |

### 3.2 Structural recovery across seeds

Transfer matrix A vs the expert prerequisite graph e1 (52 edges), edge-AUC over five seeds:
0.534 / 0.552 / 0.576 / 0.581 / 0.609 — **mean 0.567, sd 0.026**. Marginal, seed-sensitive; the
sparse graph e2 gives no significant signal.

### 3.3 Why this happens (structural explanation)

At the practice steady state the model implies, element-wise,

```
k*  ~  A phi / (A phi + Lambda)
```

so Lambda directly sets the predicted response probability per topic. Minimising the data term
therefore *forces* Lambda to absorb the observed per-topic error rate: the correlation with
difficulty is a property of the model's stationary structure, not a discovery about the data.
The genuine forgetting signal does exist in Junyi — accuracy drops from 0.751 (gaps < 1 h) to
0.589 (gaps > 30 days), topic-controlled, Spearman rho = -0.176 within topics — but it lives in
the 2.4% of interactions separated by a day or more and is dominated in the loss by the remaining
97%.

This is exactly what Theorem 1 predicts: without the excitation conditions (rest / single-concept
practice / cross practice), the parameters are not separately identifiable. Observational logs do
not provide that excitation.

## 4. Engineering

| Item | Result |
|---|---|
| Training speedup (architecture iterations) | ~400x |
| Batched solver vs per-student path | predictions bit-exact; gradients ~1e-10; ~245x faster on large data |

## 5. Methodological implication

The standard practice of validating an "interpretable" knowledge-tracing model by correlating a
learned parameter with an external property is **not sufficient evidence of recovery**: here such
a correlation (rho = 0.83) survives destroying the temporal structure of the data and is matched
by a classical BKT baseline. We recommend reporting, at minimum, (i) a shuffle control and
(ii) a simple-baseline control before claiming parameter interpretability.

## Refuted claims (do not use)

- "AUC ~0.82 on ASSISTments-2009" — came from a contaminated preprocessing path (duplicated rows,
  Xiong et al. 2016). Superseded by Section 2.
- "Lambda recovers concept difficulty (rho = 0.92), demonstrating identifiability on real data" —
  the single-run figure was 0.92, the five-seed figure 0.83, and controls C1-C4 show the
  correlation does not evidence recovery of forgetting. Superseded by Section 3.
- "Structure recovery is significant (edge-AUC 0.58, p = 0.025)" — that was a favourable seed;
  across seeds 0.567 +- 0.026, i.e. marginal. Superseded by Section 3.2.

## Reproducibility notes

- ASSISTments-2009 duplicated rows removed before all runs.
- Junyi: 25k-student subsample, seed 42, fold 0; `integrator_step = 0.25`.
- Real-time variant: node times `cumsum(log1p(dt_hours))` quantised to a 0.25 grid with strict
  monotonicity; this keeps the batched solver on a common integer grid.
