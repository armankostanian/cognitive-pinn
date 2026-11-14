"""Досчёт сбойных повторов (NaN): python rerun_failed.py
Исправленный exactkt.py пересчитывает только повторы с NaN, затем подменяет их в results_sim."""
import json, glob, os, sys, subprocess, numpy as np
from concurrent.futures import ThreadPoolExecutor

def finite(r):
    v = np.array(r["Lam"] + r["alpha"] + sum(r["A"], []) + r["beta"], float); return np.isfinite(v).all()

if len(sys.argv) > 1 and sys.argv[1] == "--one":                      # одна подгонка
    import torch, exactkt as E
    from full_sim import CONDS
    torch.set_num_threads(1)
    name, rep = sys.argv[2], int(sys.argv[3]); N, rl, tau, items, ngh, bmode = CONDS[name]
    dd = E.registered_design(); rng = np.random.default_rng(10000 + rep)
    (Y, NI, mask), eta = E.simulate(N, dd, rng, tau=tau, n_items=items); dd["P"] = torch.tensor(dd["P_learner"])
    kw = dict(fix_beta=[5.0, 4.0]) if bmode == "fix" else dict(beta_prior=(4.0, 0.5))
    mp = E.fit_marginal(Y, NI, mask, dd, random_lambda=rl, n_gh=ngh, outer=5, **kw)
    g = {k: v.detach().numpy().tolist() for k, v in mp.glob().items()}
    rec = dict(cond=name, rep=rep, Lam=g["Lam"], alpha=g["alpha"], A=g["A"], beta=g["beta"],
               tau=np.sqrt(np.exp(mp.logT.detach().numpy())).tolist() if rl else None,
               eta_rmse=float(np.sqrt(((np.array(g["eta"]) - eta) ** 2).mean())), refit=True)
    json.dump(rec, open(f"results_sim/refit_{name}_{rep:04d}.json", "w"), indent=1); sys.exit(0)

failed = []
for f in sorted(glob.glob("results_sim/*.json")):
    if os.path.basename(f).startswith("refit_"): continue
    for r in json.load(open(f)):
        if not finite(r) and not os.path.exists(f"results_sim/refit_{r['cond']}_{r['rep']:04d}.json"):
            failed.append((r["cond"], r["rep"]))
print(f"сбойных повторов к досчёту: {len(failed)}", flush=True)
cores = max(1, (os.cpu_count() or 2) - 1)
def run(job):
    c, r = job
    p = subprocess.run([sys.executable, "rerun_failed.py", "--one", c, str(r)], capture_output=True, text=True)
    print(f"{c} rep {r}: " + ("готово" if p.returncode == 0 else "ОШИБКА " + p.stderr[-300:]), flush=True)
with ThreadPoolExecutor(cores) as ex: list(ex.map(run, failed))
# подмена: заменить NaN-записи досчитанными
replaced = 0
for f in sorted(glob.glob("results_sim/*.json")):
    if os.path.basename(f).startswith("refit_"): continue
    rs = json.load(open(f)); out = []
    for r in rs:
        rf = f"results_sim/refit_{r['cond']}_{r['rep']:04d}.json"
        if not finite(r) and os.path.exists(rf): out.append(json.load(open(rf))); replaced += 1
        else: out.append(r)
    json.dump(out, open(f, "w"), indent=1)
print(f"подменено записей: {replaced}. Теперь: python summarize.py", flush=True)
