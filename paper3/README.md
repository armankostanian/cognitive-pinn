# Paper 3 — parallel exact trajectories and parameter estimation

Code, tests and raw measurements for the paper

> A. A. Kostanian, A. L. Beklaryan. *Parallel exact trajectory evaluation and parameter estimation for dissipative learning models on graphics processors*
> (in Russian; submitted to
> «Вестник ЮУрГУ. Серия: Вычислительная математика и информатика», 2026).

## Files

| File | Purpose |
|---|---|
| `scan_torch.py` | segmented prefix scan (`seg_scan`), hybrid blocked scan (`chunked_seg_scan`), memory-efficient reverse pass (`SegAffineScan`), entry point `trajectories(...)` |
| `ref_numpy.py` | NumPy reference: sequential forward pass, sequential adjoint, reference scan |
| `chunked_ref.py` | NumPy reference of the hybrid blocked algorithm |
| `tests/test_scan.py` | forward pass vs reference, reverse pass vs autograd |
| `bench.py` | synthetic benchmarks (`--exp`, `--device`, `--out`) |
| `results/` | raw measurements used in the paper |

## Run

```bash
pip install torch numpy pytest
cd paper3
pytest tests/                 # run the tests first
python bench.py --device cpu --out results_cpu.jsonl
python bench.py --device cuda --out results_gpu.jsonl
```

## Results (`results/`)

| File | Content |
|---|---|
| `synthetic_cpu_round2.jsonl`, `synthetic_gpu_round2.jsonl` | synthetic benchmarks reported in the paper (including the hybrid algorithm and block-size sweep) |
| `synthetic_cpu_round1.jsonl`, `synthetic_gpu_round1.jsonl` | first round, before the hybrid algorithm was added |
| `junyi.json` | Junyi-2015 (17,500 learners, K = 39): RK4 vs exact sequential loop vs hybrid scan |
| `device.json` | single-thread timings of one exact step and one fine-tuning iteration |
| `env.json` | hardware and software used for all measurements |

Timings depend on hardware; `env.json` records the machine on which the published numbers were
obtained. The parallel scan is **not** uniformly faster than the sequential loop: on short histories
of equal length the loop wins. The paper gives the selection rule.

Raw Junyi logs are not redistributed.
