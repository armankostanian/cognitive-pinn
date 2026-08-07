# Cognitive-PINN

**Identifiability of a dissipative knowledge-dynamics model**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reference implementation accompanying the paper *Identifiability of a dissipative
knowledge-dynamics model: exact recovery under designed excitation, degeneration on observational
data* (in preparation, 2026).

---

## Abstract

Human learning is a dissipative dynamical process: mastery accumulates through practice, decays
through forgetting, and propagates across interdependent concepts. We model it as a nonlinear
dissipative system of ODEs whose parameters are mechanistically meaningful — a concept-transfer
matrix encoding prerequisite coupling, per-concept forgetting rates, and a saturating
practice-response gain — and we study **when those parameters can actually be recovered from
data**.

We prove a structural identifiability theorem under explicit excitation conditions, with
constructive closed-form recovery for the two-concept case, and develop an L-stable semi-implicit
integrator plus a batched solver that is numerically equivalent to the per-trajectory formulation
while being two orders of magnitude faster.

The empirical picture is two-sided. Under the excitation conditions, recovery is exact. On large
observational benchmarks it is not: an apparently strong recovery is refuted by four independent
controls and traced to the stationary structure of the model.

---

## Key results

| Result | Where | Status |
|---|---|---|
| K=2 closed-form parameter recovery (error <= 1e-15) | `src/cpinn_identifiability_K2.py` | Reproducible (CPU, ~15 s) |
| K=5 prerequisite-graph recovery, F1 = 1.0, AUC = oracle | `src/cpinn_synthetic_pipeline.py` | Reproducible (CPU, ~3 min) |
| Batched solver: bit-exact vs per-student path, ~245x speedup | `src/cpinn_batched_test.py` | Reproducible |
| Clean matched AUC vs DKT/BKT (CPINN is **lower**) | `RESULTS.md` §2 | Done |
| Observational data: recovery degenerates (4 controls) | `RESULTS.md` §3 | Done |
| On-device adjoint fine-tuning with measured latency | — | In preparation |

**Read [`RESULTS.md`](RESULTS.md) before using any number from this repository.** It lists the
verified results *and* the refuted claims that earlier versions of this repository contained.

> **Two notes on what this repository does and does not show.**
> *Accuracy:* Cognitive-PINN is not state-of-the-art on response prediction — it scores below DKT
> on all clean benchmarks and below BKT on two of three. This is the deliberate cost of the
> constraints that buy identifiability.
> *Parameter recovery:* on observational logs, the recovered forgetting rates correlate with topic
> difficulty (rho = 0.83), but four controls show this is a reparameterisation of answer
> frequencies rather than recovery of forgetting — see `RESULTS.md` §3.

---

## Mathematical model

The knowledge state k(t) in [0,1]^K evolves as

```
dk/dt = A * phi(p(t)) (.) (1 - k) - Lambda (.) k + sigma (.) dW_t
```

with practice response `phi_j(p) = 1 - exp(-alpha_j * p)` and structurally factorised transfer
matrix `A = D^{1/2} (Q_prior (.) B) D^{1/2}`, where `(.)` is the element-wise product.

Proof walkthroughs: [`docs/theorem1_proof_K2.md`](docs/theorem1_proof_K2.md),
[`docs/lemma3_robustness.md`](docs/lemma3_robustness.md).

---

## Quick start

```bash
git clone https://github.com/armankostanian/cognitive-pinn.git
cd cognitive-pinn
pip install -r requirements.txt
cd src
python cpinn_identifiability_K2.py
```

Expected: parameter-recovery error <= 1e-15 at zero noise, confirming the identifiability theorem
numerically.

### Synthetic experiments (CPU)

```bash
cd src/
python cpinn_identifiability_K2.py     # closed-form recovery (Theorem 1)
python cpinn_synthetic_pipeline.py     # synthetic K=5 end-to-end
python cpinn_oracle_check.py           # sanity check with oracle parameters
```

### Batched solver

```bash
cd src/
python cpinn_batched_test.py           # equivalence + speedup test
```

The batched path integrates a whole batch with a single ODE solve on a shared integer time grid;
predictions are bit-exact against the per-student path and gradients agree to ~1e-10.

### Settings used for the reported runs

`lambda_ode = 0` (ODE-residual disabled, so batched and per-student objectives are identical),
`integrator_step = 0.25`; the structure experiments additionally set `lambda_l1 = lambda_l2 = 0`.
These are the defaults in `CPINNConfig`.

---

## Repository structure

```
cognitive-pinn/
|- README.md                       - this file
|- RESULTS.md                      - verified results + refuted claims
|- HOW_TO_RUN.md                   - run guide
|- HOW_TO_GITHUB.md                - git/github notes
|- LICENSE, requirements.txt
|
|- src/
|  |- cpinn_model.py               - model (ODE, emission, loss)
|  |- cpinn_batched.py             - batched solver (equivalent, fast)
|  |- cpinn_batched_test.py        - equivalence + speedup test
|  |- cpinn_assist09_loader.py     - per-student data format
|  |- cpinn_synthetic_pipeline.py  - synthetic K=5 pipeline
|  |- cpinn_identifiability_K2.py  - closed-form recovery validation
|  |- cpinn_oracle_check.py        - oracle sanity check
|
|- docs/
|  |- theorem1_proof_K2.md         - full proof walkthrough
|  |- lemma3_robustness.md         - perturbation analysis
|  |- research_direction.md        - current framing of the project
|
|- figures/
```

---

## Data

Experiments use public knowledge-tracing benchmarks (ASSISTments-2009/2015/2017, Junyi-2015). Raw
student logs are **not** redistributed. ASSISTments-2009 contains duplicated rows (Xiong et al.,
2016) that inflate AUC for all models; all runs here use de-duplicated data with student-level
folds shared across models and DeLong significance testing.

---

## Citation

```bibtex
@article{cognitivepinn2026,
  title   = {Identifiability of a dissipative knowledge-dynamics model:
             exact recovery under designed excitation, degeneration on
             observational data},
  author  = {Kostanian, Arman and Beklaryan, Armen},
  year    = {2026},
  note    = {in preparation}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
