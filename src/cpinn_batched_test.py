"""
cpinn_batched_test.py — доказательство, что батчевый путь численно совпадает с
одностудентным (математика модели не изменена), + замер ускорения.

Запуск:  python src/cpinn_batched_test.py
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from cpinn_model import CPINNConfig, CognitivePINN, _interp_traj
from cpinn_assist09_loader import StudentRecord, student_to_cpinn_format
import cpinn_batched as cb

torch.manual_seed(0)
np.random.seed(0)


def make_synthetic_students(n_students, K, len_min=5, len_max=40):
    """Генерируем студентов В ТОЧНОМ ФОРМАТЕ лоадера (через student_to_cpinn_format),
    чтобы тест был верен относительно реальных данных."""
    rng = np.random.default_rng(123)
    students = []
    for _ in range(n_students):
        N = int(rng.integers(len_min, len_max + 1))
        skills = rng.integers(0, K, size=N).astype(np.int64)
        resp = rng.integers(0, 2, size=N).astype(np.int64)
        rec = StudentRecord(skill_seq=skills, response_seq=resp)
        students.append(student_to_cpinn_format(rec, K))
    return students


def per_student_predictions(model, students, step):
    """Эталон: одностудентный путь (как в cpinn_model.evaluate/train)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for s in students:
            traj = model.integrate(s["t_grid"], s["k0"], s["practice_fn"])
            k_at = _interp_traj(traj, s["t_grid"], s["item_times"])
            p = model.emission(k_at, s["q_vecs"], s["item_idx"])
            preds.append(p.numpy())
    return np.concatenate(preds)


def batched_predictions(model, students, K, step, batch_size):
    model.eval()
    batches = cb.make_batches(students, K, batch_size=batch_size,
                              sort_by_length=False)  # без сортировки -> порядок сохранён
    preds = []
    with torch.no_grad():
        for batch in batches:
            traj = cb.integrate_batch(model, batch, step)
            k_at = cb._gather_k_at_items(traj, batch)
            m = batch.mask
            # восстановить порядок студентов внутри батча
            for b in range(batch.B):
                mb = m[b]
                p = model.emission(k_at[b][mb], batch.q_vecs[b][mb],
                                   batch.item_idx[b][mb])
                preds.append(p.numpy())
    return np.concatenate(preds)


def main():
    K = 8
    STEP = 0.25
    n = 60
    cfg = CPINNConfig(K=K, rank=K, integrator="rk4", integrator_step=STEP,
                      lambda_ode=0.0)  # data-term изолирован
    model = CognitivePINN(cfg, n_items=K, q_prior=torch.ones(K, K))

    students = make_synthetic_students(n, K)

    print("=== 1. Численная эквивалентность (одинаковые веса) ===")
    p_ref = per_student_predictions(model, students, STEP)
    p_bat = batched_predictions(model, students, K, STEP, batch_size=16)
    assert p_ref.shape == p_bat.shape, (p_ref.shape, p_bat.shape)
    max_abs = float(np.max(np.abs(p_ref - p_bat)))
    mean_abs = float(np.mean(np.abs(p_ref - p_bat)))
    print(f"  предсказаний сравнено: {p_ref.shape[0]}")
    print(f"  max |Δp| = {max_abs:.3e}")
    print(f"  mean|Δp| = {mean_abs:.3e}")
    ok = max_abs < 1e-5
    print(f"  -> {'СОВПАДАЕТ (математика не изменена)' if ok else 'РАСХОЖДЕНИЕ!'}")

    print("\n=== 2. Эквивалентность градиентов BCE по A ===")
    # один backward по объединённому BCE обоими путями -> сравнить grad A
    def grad_A_per_student():
        model.zero_grad()
        import torch.nn.functional as F
        losses = []
        for s in students:
            traj = model.integrate(s["t_grid"], s["k0"], s["practice_fn"])
            k_at = _interp_traj(traj, s["t_grid"], s["item_times"])
            p = model.emission(k_at, s["q_vecs"], s["item_idx"]).clamp(1e-6, 1-1e-6)
            losses.append(F.binary_cross_entropy(p, s["y_true"], reduction="sum"))
        total_items = sum(s["y_true"].numel() for s in students)
        (torch.stack(losses).sum() / total_items).backward()
        return model.ode_func.U_raw.grad.clone()

    def grad_A_batched():
        model.zero_grad()
        batches = cb.make_batches(students, K, batch_size=16, sort_by_length=False)
        import torch.nn.functional as F
        num, den = 0.0, 0
        parts = []
        for batch in batches:
            traj = cb.integrate_batch(model, batch, STEP)
            k_at = cb._gather_k_at_items(traj, batch)
            mm = batch.mask
            p = model.emission(k_at[mm], batch.q_vecs[mm],
                               batch.item_idx[mm]).clamp(1e-6, 1-1e-6)
            parts.append(F.binary_cross_entropy(p, batch.y_true[mm], reduction="sum"))
            den += int(mm.sum())
        (torch.stack(parts).sum() / den).backward()
        return model.ode_func.U_raw.grad.clone()

    g1 = grad_A_per_student()
    g2 = grad_A_batched()
    gmax = float((g1 - g2).abs().max())
    print(f"  max |Δ grad U_raw| = {gmax:.3e}  -> "
          f"{'СОВПАДАЕТ' if gmax < 1e-5 else 'РАСХОЖДЕНИЕ!'}")

    print("\n=== 3. Замер скорости (CPU, sandbox) ===")
    big = make_synthetic_students(300, K, len_min=20, len_max=60)
    t0 = time.time()
    _ = per_student_predictions(model, big, STEP)
    t_ref = time.time() - t0
    t0 = time.time()
    _ = batched_predictions(model, big, K, STEP, batch_size=64)
    t_bat = time.time() - t0
    print(f"  одностудентный: {t_ref:.2f} c")
    print(f"  батчевый(64):   {t_bat:.2f} c")
    print(f"  ускорение CPU:  ×{t_ref / max(t_bat,1e-9):.1f} "
          f"(на GPU выигрыш кратно больше — там и узкое место)")

    print("\nИТОГ:", "OK" if (ok and gmax < 1e-5) else "ПРОВЕРИТЬ")


if __name__ == "__main__":
    main()
