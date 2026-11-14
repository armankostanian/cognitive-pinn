"""Одна порция моделирования: python full_sim.py <условие> <rep_start> <rep_end>
Результат: results_sim/<условие>_<rep_start>.json (возобновляемо)."""
import json, os, sys, numpy as np, torch, exactkt as E
torch.set_num_threads(1)
CONDS = {  # имя: (N, иерархическая, tau, задач в замере, узлов квадратуры, beta: fix|prior)
    "pop_tau0_i5":          (160, False, 0.0, 5, 7, "fix"),
    "pop_tau02_i5":         (160, False, 0.2, 5, 7, "fix"),
    "hier_tau02_i5":        (160, True,  0.2, 5, 5, "fix"),
    "hier_tau02_i10":       (160, True,  0.2, 10, 5, "fix"),
    "hier_tau02_i5_bprior": (160, True,  0.2, 5, 5, "prior"),
    "hier_tau02_i5_N320":   (320, True,  0.2, 5, 5, "fix"),
}
if __name__ == "__main__":
    name, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    N, rl, tau, items, ngh, bmode = CONDS[name]
    os.makedirs("results_sim", exist_ok=True); path = f"results_sim/{name}_{a:04d}.json"
    done = json.load(open(path)) if os.path.exists(path) else []; have = {r["rep"] for r in done}
    for r in range(a, b):
        if r in have: continue
        dd = E.registered_design(); rng = np.random.default_rng(10000 + r)
        (Y, NI, mask), eta = E.simulate(N, dd, rng, tau=tau, n_items=items); dd["P"] = torch.tensor(dd["P_learner"])
        kw = dict(fix_beta=[5.0, 4.0]) if bmode == "fix" else dict(beta_prior=(4.0, 0.5))
        mp = E.fit_marginal(Y, NI, mask, dd, random_lambda=rl, n_gh=ngh, outer=5, **kw)
        g = {k: v.detach().numpy().tolist() for k, v in mp.glob().items()}
        done.append(dict(cond=name, rep=r, Lam=g["Lam"], alpha=g["alpha"], A=g["A"], beta=g["beta"],
                         tau=np.sqrt(np.exp(mp.logT.detach().numpy())).tolist() if rl else None,
                         eta_rmse=float(np.sqrt(((np.array(g["eta"]) - eta) ** 2).mean()))))
        json.dump(done, open(path, "w"), indent=1)
        print(f"{name} rep {r}: Lam {np.round(g['Lam'], 4)}", flush=True)
