# Paper 1 — control experiments

Controls for paper 1 (*Identifiability of a dissipative knowledge-dynamics model …*, arXiv:2610.09889).
Configuration of the additional controls (from `REPORT_extra_controls_ru.md`, §2): Junyi-2015,
K = 39 topics, fold 0 (17,500 / 2,500 / 5,000 learners), rank 25, training seeds 0–4, 25 epochs,
lr 3e-3, `lambda_ode = 0`, `l1 = l2 = 0`, integrator step 0.25.

## Files in this directory

| Path | Content |
|---|---|
| `REPORT_extra_controls_ru.md` | report (in Russian) on the additional controls E3, E1, E1b, E2: what was run, numbers, what may and may not be claimed |
| `original_controls/exp1_shuffle.py` | shuffling control: interaction order is permuted within each learner, the model is retrained, Λ is compared with topic difficulty |
| `original_controls/exp2_seeds_structure.py` | reproducibility over seeds of the structural result (edge AUC of A against the expert prerequisite graph) |
| `original_controls/exp3_bkt_params.py` | baseline control: standard BKT parameters vs. topic difficulty on the same Junyi fold 0 |
| `original_controls/eval_structure.py` | edge AUC / AP of the recovered transfer matrix A against the expert graph |
| `original_controls/mac_common.py`, `cpinn_clean_loader.py`, `datasets.py`, `bkt_simple.py` | shared helpers (device selection, fold loading, training with a fixed number of epochs; clean-data loader; self-contained BKT) |
| `original_controls/topic_difficulty.npy`, `topic_count.npy` | small per-topic arrays |

## Additional controls described in the report (scripts not yet in the repository)

| Control | Question | Section of the report |
|---|---|---|
| E3 | ablation: one shared Λ instead of per-topic Λ | §3 |
| E1 | positive control on a synthetic cohort (parameters as specified) | §4.1 |
| E1b | positive control with parameters at which forgetting is observable | §4.2 |
| E2 | isolating the mechanism on the real Junyi schedule | §5 |

The scripts for these (`e3_*.py`, `e1_*.py`, `e1b_select.py`, `e2_*.py`, `build_results.py`,
`lambda_seed_repro_sweep.py`) and `results_extra.json` are **not** in this directory yet; the numbers
are available only in the report.
