"""
exp1_shuffle.py — ГЛАВНЫЙ контроль тривиальности Λ-результата.

Вопрос: Λ читает ВРЕМЕННУЮ динамику ответов или просто впитывает частоту ошибок
по темам? Перемешиваем порядок взаимодействий ВНУТРИ каждого студента (частоты
тем/ответов сохранены, временная структура уничтожена), переобучаем и смотрим,
сохранится ли rho(Λ, difficulty) ~ 0.83.

Интерпретация:
  rho_shuffled << 0.83 (например < ~0.4)  -> Λ читает динамику: сильный аргумент.
  rho_shuffled ~ 0.83                     -> Λ впитывает частоты: честно ослабляем claim.

Запуск (из pipeline/):  python exp1_shuffle.py
Выход: results/Lambda_shuffled.npy, results/A_shuffled.npy, печать сравнения.
"""
from __future__ import annotations
import os, shutil
import numpy as np
import pandas as pd
import mac_common as mc

SEED_SHUFFLE = 42
SEED_MODEL = 0
REF_RHO = 0.834          # среднее по 5 сидам на оригинальных данных (25k)


def make_shuffled_dataset(src="clean/junyi", dst="clean/junyi_shuffled",
                          seed=SEED_SHUFFLE):
    """Копия датасета с перемешанным порядком внутри каждого студента."""
    if os.path.exists(os.path.join(dst, "data.csv")):
        print(f"[shuffle] {dst} уже существует — использую его")
        return
    os.makedirs(dst, exist_ok=True)
    data = pd.read_csv(os.path.join(src, "data.csv"))
    rng = np.random.default_rng(seed)

    data = data.sort_values(["student", "step"])
    parts = []
    for _, g in data.groupby("student", sort=True):
        idx = rng.permutation(len(g))
        g = g.iloc[idx].copy()
        g["step"] = np.arange(len(g))
        parts.append(g)
    out = pd.concat(parts, ignore_index=True).sort_values(["student", "step"])
    out.to_csv(os.path.join(dst, "data.csv"), index=False)
    shutil.copy(os.path.join(src, "folds.csv"), os.path.join(dst, "folds.csv"))
    for extra in ("skill_topics.csv", "meta.txt"):
        p = os.path.join(src, extra)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(dst, extra))
    # sanity: частоты не изменились
    a = data.groupby("skill")["correct"].mean().sort_index()
    b = out.groupby("skill")["correct"].mean().sort_index()
    assert np.allclose(a.values, b.values), "per-skill частоты изменились — баг"
    print(f"[shuffle] создан {dst}: {len(out)} взаимодействий, "
          f"per-skill частоты идентичны оригиналу")


def main():
    device = mc.pick_device()
    print(f"[device] {device}")
    make_shuffled_dataset()

    bundle = mc.load_junyi_fold0("junyi_shuffled")
    K = bundle.K
    print(f"[data] junyi_shuffled fold0: K={K}, train={len(bundle.train)}, "
          f"val={len(bundle.val)}")

    train_b = mc.build_batches(bundle.train, K, device)
    val_b = mc.build_batches(bundle.val, K, device)

    model = mc.make_model(K, seed=SEED_MODEL, device=device)
    res = mc.train_fixed(model, train_b, val_b, device, tag="shuffle")

    A, Lam = mc.extract_params(model)
    os.makedirs("results", exist_ok=True)
    np.save("results/A_shuffled.npy", A)
    np.save("results/Lambda_shuffled.npy", Lam)

    st = mc.spearman_with_partial(Lam)
    print("\n================ ИТОГ ЭКСПЕРИМЕНТА 1 (SHUFFLE) ================")
    print(f"val_auc (shuffled data)        = {res['val_auc']:.4f}")
    print(f"rho(Lambda_shuffled, difficulty) = {st['rho']:.3f}  (p={st['p']:.2g})")
    print(f"partial rho | count              = {st['partial_rho']:.3f}  (p={st['partial_p']:.2g})")
    print(f"reference (оригинальные данные)  = {REF_RHO:.3f} (среднее по 5 сидам)")
    verdict = ("Λ ЧИТАЕТ ДИНАМИКУ (rho упал существенно) — сильный аргумент в статью"
               if st['rho'] < 0.55 else
               "Λ ВПИТЫВАЕТ ЧАСТОТЫ (rho сохранился) — ослабляем claim в статье")
    print(f"ВЕРДИКТ: {verdict}")
    print("Файлы: results/Lambda_shuffled.npy, results/A_shuffled.npy")


if __name__ == "__main__":
    main()
