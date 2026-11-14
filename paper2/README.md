# Paper 2 — exact discretisation and an identifiability criterion

Code and results for the paper

> A. Kostanian, A. Beklaryan. *Exact discretisation and a criterion for parameter identifiability in
> input-driven dissipative models of learning* (under review, Journal of Mathematical Psychology).

## Files

| File | Purpose |
|---|---|
| `exactkt.py` | model (exact discretisation) and likelihood; Laplace-EM estimation |
| `verify_theory.py`, `verify_emission.py`, `verify_forms.py`, `verify_hetero.py` | numerical verification of the theorems |
| `full_sim.py`, `run_all.py`, `rerun_failed.py`, `summarize.py` | simulation experiment (run, parallel launcher, re-fit of failed runs, summary) |
| `scan_bench.py` | parallel prefix scan for the affine recursion vs. sequential exact step and RK4 |
| `results_sim/` | raw per-run results of the simulation experiment (JSON) |
| `summary.csv` | aggregated results |

## Run

```bash
pip install numpy scipy torch
cd paper2
python verify_theory.py                  # numerical checks of the theory
python run_all.py --smoke                # quick check of the simulation (one repetition per condition)
python run_all.py                        # full simulation, uses all cores but one
python rerun_failed.py                   # optional: re-fit runs that failed to converge
python summarize.py                      # prints the table and writes summary.csv
```
