"""
exp2_seeds_structure.py — воспроизводимость СТРУКТУРНОГО результата по сидам.

Λ уже прошла 5-сидовый контроль (rho = 0.83 +- 0.01). Теперь тот же контроль
для edge-AUC структуры A против экспертного графа e1: фиксируем честный
диапазон для статьи (заявлено 0.581 на одном прогоне, p=0.025).

Запуск (из pipeline/):  python exp2_seeds_structure.py
Выход: results/A_seed{0..4}.npy, results/Lambda_seed{0..4}.npy, таблица.
Время: 5 обучений по 25 эпох — на Mac это самый долгий скрипт (можно на ночь).
"""
from __future__ import annotations
import os
import numpy as np
import mac_common as mc

SEEDS = [0, 1, 2, 3, 4]
GRAPH_E1 = "clean/junyi/kc_prereq_graph_e1.npy"
GRAPH_E2 = "clean/junyi/kc_prereq_graph_e2.npy"


def main():
    device = mc.pick_device()
    print(f"[device] {device}")
    bundle = mc.load_junyi_fold0("junyi")
    K = bundle.K
    print(f"[data] junyi fold0: K={K}, train={len(bundle.train)}, val={len(bundle.val)}")

    train_b = mc.build_batches(bundle.train, K, device)
    val_b = mc.build_batches(bundle.val, K, device)

    os.makedirs("results", exist_ok=True)
    rows = []
    for seed in SEEDS:
        model = mc.make_model(K, seed=seed, device=device)
        res = mc.train_fixed(model, train_b, val_b, device, tag=f"seed{seed}")
        A, Lam = mc.extract_params(model)
        np.save(f"results/A_seed{seed}.npy", A)
        np.save(f"results/Lambda_seed{seed}.npy", Lam)
        row = {"seed": seed, "val_auc": res["val_auc"],
               "edge_auc_e1": mc.edge_auc(A, GRAPH_E1)}
        if os.path.exists(GRAPH_E2):
            row["edge_auc_e2"] = mc.edge_auc(A, GRAPH_E2)
        st = mc.spearman_with_partial(Lam)
        row["rho_lambda"] = st["rho"]
        rows.append(row)
        print(f"[seed {seed}] val_auc={row['val_auc']:.4f}  "
              f"edge_auc_e1={row['edge_auc_e1']:.4f}  rho_lambda={row['rho_lambda']:.3f}")

    print("\n================ ИТОГ ЭКСПЕРИМЕНТА 2 (СТРУКТУРА ПО СИДАМ) ================")
    hdr = "seed | val_auc | edge_auc_e1" + (" | edge_auc_e2" if "edge_auc_e2" in rows[0] else "") + " | rho_lambda"
    print(hdr)
    for r in rows:
        line = f"{r['seed']:4d} | {r['val_auc']:.4f}  | {r['edge_auc_e1']:.4f}     "
        if "edge_auc_e2" in r:
            line += f" | {r['edge_auc_e2']:.4f}     "
        line += f" | {r['rho_lambda']:.3f}"
        print(line)
    e1 = np.array([r["edge_auc_e1"] for r in rows])
    rl = np.array([r["rho_lambda"] for r in rows])
    print(f"\nedge_auc_e1: mean={e1.mean():.4f}  sd={e1.std():.4f}  "
          f"min={e1.min():.4f}  max={e1.max():.4f}")
    print(f"rho_lambda : mean={rl.mean():.3f}  sd={rl.std():.3f}")
    print("Файлы: results/A_seed*.npy, results/Lambda_seed*.npy")
    import csv
    with open("results/seeds_structure_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("Таблица: results/seeds_structure_table.csv")


if __name__ == "__main__":
    main()
