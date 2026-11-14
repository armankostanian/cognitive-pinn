"""
exp3_bkt_params.py — baseline-контроль: параметры BKT против сложности тем.

Вопрос: наш claim уникальности Λ-результата держится, только если ПРОСТАЯ
классическая модель не даёт такой же корреляции «бесплатно». Обучаем стандартный
BKT (EM, per-skill) на том же Junyi fold 0 и считаем Spearman каждого параметра
(L0, T=learn, S=slip, G=guess) со сложностью тем.

Замечание: slip/guess в реализации капированы 0.30 (стандартная практика против
инверсии AUC) — при интерпретации корреляций это учитывать (связки на капе).

Запуск (из pipeline/):  python exp3_bkt_params.py     (только CPU, ~30-60 мин)
Выход: results/bkt_params.csv + таблица корреляций.
"""
from __future__ import annotations
import os, time
import numpy as np
from scipy.stats import spearmanr, rankdata, pearsonr

import datasets
from bkt_simple import fit_skill, gather_by_skill


def partial_spearman(x, y, z):
    def resid(a, b):
        b1 = np.c_[np.ones_like(b), b]
        beta = np.linalg.lstsq(b1, a, rcond=None)[0]
        return a - b1 @ beta
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    return pearsonr(resid(rx, rz), resid(ry, rz))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    diff = np.load(os.path.join(here, "topic_difficulty.npy"))
    cnt = np.load(os.path.join(here, "topic_count.npy"))

    folds = datasets.load_folds("junyi", "clean")
    f0 = folds[0]
    K = f0["K"]
    train = f0["train"] + f0["val"]          # BKT не использует val -> отдать в train
    print(f"[data] junyi fold0: K={K}, train_students={len(train)}")

    by_skill = gather_by_skill(train)
    L0 = np.full(K, np.nan); T = np.full(K, np.nan)
    S = np.full(K, np.nan); G = np.full(K, np.nan)
    t0 = time.time()
    for k in range(K):
        seqs = by_skill.get(k, [])
        if len(seqs) < 5:
            print(f"[skill {k}] слишком мало последовательностей ({len(seqs)}) — пропуск")
            continue
        p = fit_skill(seqs)
        L0[k], T[k], S[k], G[k] = p
        if (k + 1) % 10 == 0:
            print(f"[bkt] {k+1}/{K} skills, {time.time()-t0:.0f} c")
    print(f"[bkt] EM завершён за {time.time()-t0:.0f} c")

    os.makedirs("results", exist_ok=True)
    import csv
    with open("results/bkt_params.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["skill", "L0", "T_learn", "S_slip", "G_guess",
                                       "difficulty", "count"])
        for k in range(K):
            w.writerow([k, L0[k], T[k], S[k], G[k], diff[k], int(cnt[k])])

    print("\n================ ИТОГ ЭКСПЕРИМЕНТА 3 (BKT-КОНТРОЛЬ) ================")
    print("параметр  | rho vs difficulty |    p     | partial|count |    p")
    ref = {"CPINN Lambda (5 сидов)": (0.834, None, 0.775, None)}
    rows = {"L0 (prior)": L0, "T (learn)": T, "S (slip)": S, "G (guess)": G}
    for name, v in rows.items():
        m = ~np.isnan(v)
        if m.sum() < 5:
            print(f"{name:9s} | недостаточно данных"); continue
        rho, p = spearmanr(v[m], diff[m])
        pr, pp = partial_spearman(v[m], diff[m], cnt[m])
        print(f"{name:9s} |      {rho:+.3f}      | {p:.2g} |    {pr:+.3f}    | {pp:.2g}")
    print(f"reference: CPINN Lambda rho=+0.834 (partial +0.775), 5 сидов")
    print("Интерпретация: если |rho| BKT-параметров сопоставим с 0.83 — уникальность")
    print("Λ-результата ослабляем; если заметно ниже — усиливаем (у нас есть сравнение).")
    print("Файл: results/bkt_params.csv")


if __name__ == "__main__":
    main()
