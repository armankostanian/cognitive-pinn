"""Запуск всего моделирования на всех ядрах: python run_all.py [--smoke]
--smoke: по 1 повтору каждого условия (проверка, что всё работает; ~10-30 мин)."""
import os, sys, subprocess
from concurrent.futures import ThreadPoolExecutor
from full_sim import CONDS
smoke = "--smoke" in sys.argv
REPS = {name: (1 if smoke else (200 if name.startswith("pop") else 100)) for name in CONDS}
CHUNK = 1 if smoke else 10
jobs = [(n, s, min(s + CHUNK, REPS[n])) for n in CONDS for s in range(0, REPS[n], CHUNK)]
cores = max(1, (os.cpu_count() or 2) - 1)
print(f"порций: {len(jobs)}, параллельно: {cores}", flush=True)
def run(j):
    n, a, b = j
    r = subprocess.run([sys.executable, "full_sim.py", n, str(a), str(b)], capture_output=True, text=True)
    print(f"готово: {n} {a}-{b}" + ("" if r.returncode == 0 else f"  ОШИБКА: {r.stderr[-300:]}"), flush=True)
with ThreadPoolExecutor(cores) as ex: list(ex.map(run, jobs))
print("ВСЁ. Теперь: python summarize.py", flush=True)
