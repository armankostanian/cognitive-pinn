"""Сводка: python summarize.py -> таблица и summary.csv (прислать)."""
import json, glob, csv, os, numpy as np
from collections import defaultdict
T = dict(L=[0.05, 0.07], a=[0.12, 0.10], A=[5.0, 4.0])
R = defaultdict(list)
for f in glob.glob("results_sim/*.json"):
    if os.path.basename(f).startswith("refit_"): continue
    for r in json.load(open(f)): R[r["cond"]].append(r)
rows = []
for c, rs in sorted(R.items()):
    L = np.array([r["Lam"] for r in rs], float); a = np.array([r["alpha"] for r in rs], float)
    A = np.array([[r["A"][0][0], r["A"][1][1]] for r in rs], float)
    ok = np.isfinite(L).all(1) & np.isfinite(a).all(1); n = int(ok.sum())
    row = dict(cond=c, reps=len(rs), failed=int((~ok).sum()))
    for i in range(2):
        row[f"Lam{i+1}_bias%"] = 100 * (L[ok, i].mean() / T["L"][i] - 1)
        row[f"Lam{i+1}_mcse%"] = 100 * L[ok, i].std() / T["L"][i] / np.sqrt(n)
        row[f"Lam{i+1}_sd%"] = 100 * L[ok, i].std() / T["L"][i]
        row[f"alpha{i+1}_medbias%"] = 100 * (np.median(a[ok, i]) / T["a"][i] - 1)
        row[f"alpha{i+1}_iqr%"] = 100 * np.subtract(*np.percentile(a[ok, i], [75, 25])) / T["a"][i]
        row[f"A{i+1}{i+1}_medbias%"] = 100 * (np.median(A[ok, i]) / T["A"][i] - 1)
    taus = [r["tau"] for r, o in zip(rs, ok) if o and r.get("tau")]
    if taus: row["tau1_mean"], row["tau2_mean"] = map(float, np.mean(taus, 0))
    betas = np.array([r["beta"] for r, o in zip(rs, ok) if o]); row["beta1_mean"], row["beta2_mean"] = map(float, betas.mean(0))
    rows.append(row)
keys = list(dict.fromkeys(k for r in rows for k in r))
with open("summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
for r in rows: print({k: (round(float(v), 2) if isinstance(v, float) else v) for k, v in r.items()})
print("\nsummary.csv готов — пришли его и архив папки results_sim.")
