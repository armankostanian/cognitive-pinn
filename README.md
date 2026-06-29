# Cognitive-PINN

**Physics-Informed Modeling of Student Knowledge Dynamics with Identifiable Parameters**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reference implementation accompanying the paper *Cognitive-PINN: Physics-Informed Modeling of Student Knowledge Dynamics with Identifiable Parameters* (in preparation, 2026).

---

## Abstract

Deep knowledge-tracing models achieve strong predictive performance on student-response prediction, yet their hidden-state dynamics are not physically interpretable: forgetting is modelled *ad hoc* and prerequisite structure is implicit. We introduce **Cognitive-PINN**, a physics-informed framework that formulates the evolution of a student's knowledge state as a dissipative ordinary differential equation whose parameters carry an explicit pedagogical meaning (a concept-transfer matrix, per-concept forgetting rates, and a practice-response gain).

The contribution of this work is **methodological and theoretical, not predictive accuracy**. We prove a **structural identifiability theorem** for the resulting inverse problem under explicit conditions on the practice schedule, accompanied by stability and robustness lemmas, and we develop an **L-stable numerical integrator** and a memory-efficient adjoint scheme for parameter estimation. On clean benchmark data the model is intentionally *less* accurate than DKT/BKT — a documented trade-off paid for interpretability and identifiability. On real data (Junyi), the recovered forgetting rates correlate strongly with independently measured topic difficulty (Spearman rho = 0.92), demonstrating that the parameters are measurable and meaningful.

---

## Key results (honest summary)

| Result | Where | Status |
|---|---|---|
| Closed-form parameter recovery on synthetic K=2 (error <= 1e-15) | `src/cpinn_identifiability_K2.py` | Reproducible (CPU, ~15 s) |
| Synthetic K=5: prerequisite-graph recovery F1 = 1.0, AUC = oracle | `src/cpinn_synthetic_pipeline.py` | Reproducible (CPU, ~3 min) |
| Batched solver: bit-exact vs per-student path, ~245x speedup | `src/cpinn_batched_test.py` | Reproducible |
| Clean matched AUC vs DKT/BKT (CPINN is **lower** — see note) | external pipeline | Reproducible |
| Junyi: forgetting Lambda vs topic difficulty, rho = 0.92 (partial 0.86) | `RESULTS.md`, `figures/fig_lambda_difficulty.png` | Done |
| Junyi: structure A vs expert graph, edge-AUC 0.58 (p = 0.025) | `RESULTS.md`, `figures/fig_structure_significance.png` | Done (weak but significant) |
| On-device adjoint fine-tuning with measured latency | — | In preparation |

See **[`RESULTS.md`](RESULTS.md)** for the full validation summary, including the accuracy trade-off.

> **Note on accuracy.** Cognitive-PINN is *not* state-of-the-art on response prediction. On clean, contamination-controlled benchmarks it scores below DKT on all datasets and below BKT on two of three. This is a deliberate consequence of the structural constraints that make the model interpretable and its parameters identifiable. The value of the model lies in those guarantees and in the external validation of its parameters — not in leaderboard accuracy.

---

## Mathematical model

The student's knowledge state k(t) in [0,1]^K evolves as

```
dk/dt = A * phi(p(t)) (.) (1 - k) - Lambda (.) k + sigma (.) dW_t
```

with practice response `phi_j(p) = 1 - exp(-alpha_j * p)` and structurally factorised transfer
matrix `A = D^{1/2} (Q_prior (.) B) D^{1/2}`, where `(.)` is the element-wise product.

Parameter meaning: **A** — concept-to-concept transfer (prerequisite structure);
**Lambda** — per-concept forgetting rates; **alpha** — practice-response gain.
The identifiability theorem, stability/robustness lemmas, and the L-stable integrator are
developed in the paper; proof walkthroughs are in [`docs/`](docs/).

---

## Quick start

```bash
git clone https://github.com/armankostanian/cognitive-pinn.git
cd cognitive-pinn
pip install -r requirements.txt
cd src
python cpinn_identifiability_K2.py
```

You should see a parameter-recovery error <= 1e-15 at zero noise, confirming the
identifiability theorem numerically.

### Synthetic experiments (CPU, no GPU needed)

```bash
cd src/
python cpinn_identifiability_K2.py     # closed-form recovery (Theorem 1)
python cpinn_synthetic_pipeline.py     # synthetic K=5 end-to-end
python cpinn_oracle_check.py           # sanity check with oracle parameters
```

### Batched solver (large datasets)

For large cohorts the per-student training loop is the bottleneck. `src/cpinn_batched.py`
integrates the whole batch with a single ODE solve on a shared integer time grid; it is
numerically equivalent to the per-student path (predictions bit-exact, gradients to ~1e-10)
and ~245x faster on large data. Verify the equivalence:

```bash
cd src/
python cpinn_batched_test.py
```

> **Reproducibility note.** The clean real-data experiments and the Junyi validation were run
> with `lambda_ode = 0` (the ODE-residual regulariser disabled) so that the batched and
> per-student objectives are identical; for the structure experiment the L1/L2 penalties on
> **B** were also disabled (ablation). See `RESULTS.md`.

---

## Repository structure

```
cognitive-pinn/
|- README.md                       - this file
|- RESULTS.md                      - validation results (clean numbers + figures)
|- HOW_TO_RUN.md                   - run guide
|- HOW_TO_GITHUB.md                - git/github crash course
|- LICENSE                         - MIT
|- requirements.txt                - Python dependencies
|
|- src/
|  |- cpinn_model.py               - Cognitive-PINN PyTorch model
|  |- cpinn_batched.py             - batched ODE solver (equivalent, fast)
|  |- cpinn_batched_test.py        - equivalence + speedup test
|  |- cpinn_assist09_loader.py     - per-student data format
|  |- cpinn_synthetic_pipeline.py  - synthetic K=5 end-to-end pipeline
|  |- cpinn_identifiability_K2.py  - closed-form recovery validation
|  |- cpinn_oracle_check.py        - sanity check with ground-truth ODE params
|
|- figures/
|  |- architecture.png
|  |- identifiability_K2.png
|  |- cpinn_A_compare.png
|  |- cpinn_synthetic_training.png
|  |- fig_lambda_difficulty.png       - Lambda vs topic difficulty (rho=0.92)
|  |- fig_structure_significance.png  - A vs expert graph (edge-AUC 0.58, p=0.025)
|
|- docs/
   |- theorem1_proof_K2.md         - full step-by-step proof
   |- lemma3_robustness.md         - perturbation analysis
   |- cognitive_pinn_proposal.md   - research direction (current framing)
```

---

## Requirements

- Python >= 3.10, PyTorch >= 2.0, torchdiffeq >= 0.2.5
- scikit-learn >= 1.3, numpy, scipy, matplotlib, pandas

Full list in [`requirements.txt`](requirements.txt). For GPU experiments: NVIDIA GPU with >= 8 GB memory.

---

## Data

The real-data experiments use standard public knowledge-tracing benchmarks
(ASSISTments-2009/2015/2017 and Junyi-2015). We do **not** redistribute raw student logs.
Note that ASSISTments-2009 contains duplicated rows (Xiong et al., 2016) that inflate AUC for
all models; our experiments use a de-duplicated, contamination-controlled preprocessing with
student-level folds shared across models and DeLong significance testing.

---

## Citation

```bibtex
@article{cognitivepinn2026,
  title   = {Cognitive-PINN: Physics-Informed Modeling of Student
             Knowledge Dynamics with Identifiable Parameters},
  author  = {Kostanian, Arman and Beklaryan, Armen},
  year    = {2026},
  note    = {in preparation}
}
```

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Contact

For questions about the paper or the code, please open a GitHub issue.
