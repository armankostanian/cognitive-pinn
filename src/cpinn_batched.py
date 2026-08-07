"""
cpinn_batched.py — батчевый путь обучения/инференса Cognitive-PINN.

Зачем
-----
Медленный путь в cpinn_model.py (`train_one_epoch`, `evaluate`) интегрирует
КАЖДОГО студента отдельным вызовом odeint и делает optimizer.step() на одного
студента (SGD с batch=1). На Junyi (~25k студентов) это ~6 ч/эпоху и GPU на 18%.

Ключевое наблюдение, делающее батчинг возможным БЕЗ изменения математики:
псевдовремя в лоадере — это ИНДЕКС взаимодействия (t_i = i, dt=1). Значит у всех
студентов общая ЦЕЛОЧИСЛЕННАЯ сетка времени. Поэтому можно проинтегрировать
состояние формы (B, K) одним вызовом odeint по общей сетке [0..T_max], а в лоссе
читать k только в моменты item_times каждого студента (с маской на паддинг).

Эквивалентность математики
---------------------------
1) Правая часть ОДУ: A @ phi  ==  phi @ A.T поэлементно. Для k формы (B,K) и
   practice формы (B,K) формула (phi @ A.T) * (1-k) - Lambda*k даёт ровно те же
   числа, что одностудентный путь, просто разом для всего батча.
2) rk4 с фиксированным шагом локален: продолжение сетки за пределы N_i не меняет
   значения k(t) при t<=N_i. Поэтому интеграция короткого студента до T_max не
   портит его траекторию на «своём» участке.
3) Эмиссия и интерполяция переиспользуются как есть (тот же IRTLikeEmission,
   та же линейная интерполяция по времени).

Тест cpinn_batched_test.py доказывает совпадение траекторий и BCE с одностудентным
путём до ~1e-5.
"""

from __future__ import annotations

from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchdiffeq import odeint
from sklearn.metrics import roc_auc_score, brier_score_loss


# =============================================================================
# 1. Батчевая правая часть: одно изменение vs CognitiveODEFunc.forward
# =============================================================================
#
# В cpinn_model.CognitiveODEFunc.forward строка
#     learning = (self.A @ phi) * (1.0 - k)
# работает только для phi/k формы (K,). Чтобы тот же модуль интегрировал (B,K),
# нужна форма-агностичная запись. Делаем это БЕЗ копирования параметров —
# просто оборачиваем существующий ode_func и подменяем правило умножения.
#
# Математически (phi @ A.T) == (A @ phi) для вектора, поэтому одностудентный путь
# тоже можно было бы перевести на эту запись без изменения чисел (см. тест).


class BatchedODEWrapper(torch.nn.Module):
    """Оборачивает уже существующий CognitiveODEFunc и считает RHS для k формы
    (B, K). Использует ТЕ ЖЕ параметры (A, Lambda, alpha) — ничего не копируется,
    градиенты текут в исходный ode_func."""

    def __init__(self, ode_func):
        super().__init__()
        self.ode_func = ode_func          # ссылка, не копия
        self._practice = None             # closure: t(scalar) -> (B, K)

    def set_practice(self, practice_fn):
        self._practice = practice_fn

    def forward(self, t: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        of = self.ode_func
        p = self._practice(t)                                  # (B, K)
        phi = 1.0 - torch.exp(-of.alpha.unsqueeze(0) * p)      # (B, K)
        learning = (phi @ of.A.T) * (1.0 - k)                  # (B, K)
        forgetting = of.Lambda.unsqueeze(0) * k               # (B, K)
        return learning - forgetting


# =============================================================================
# 2. Коллация списка студентов -> один батч на общей целочисленной сетке
# =============================================================================

class StudentBatch:
    """Контейнер одного батча студентов на общей сетке t_grid=[0..T_max]."""

    __slots__ = ("t_grid", "k0", "pulse", "item_step", "q_vecs",
                 "item_idx", "y_true", "mask", "B", "T", "K")

    def __init__(self, students: List[Dict[str, torch.Tensor]], K: int):
        B = len(students)
        # длина истории = число item'ов; t_grid студента имеет N+1 точек
        lengths = [s["item_times"].shape[0] for s in students]
        L_max = max(lengths)
        T_max = max(int(s["t_grid"][-1].item()) for s in students)   # = max N_i
        T = T_max + 1

        self.B, self.T, self.K = B, T, K
        self.t_grid = torch.arange(0, T, dtype=torch.float32)        # (T,)
        self.k0 = torch.full((B, K), 0.1, dtype=torch.float32)        # (B, K)

        # practice как плотный тензор (T, B, K): для шага сетки j берём pulse[j]
        pulse = torch.zeros(T, B, K, dtype=torch.float32)
        # паддинг item'ов
        item_step = torch.zeros(B, L_max, dtype=torch.long)          # индекс в t_grid
        q_vecs = torch.zeros(B, L_max, K, dtype=torch.float32)
        item_idx = torch.zeros(B, L_max, dtype=torch.long)
        y_true = torch.zeros(B, L_max, dtype=torch.float32)
        mask = torch.zeros(B, L_max, dtype=torch.bool)

        for b, s in enumerate(students):
            Ni = lengths[b]
            tg_b = s["t_grid"]                                       # (Ni+1,)
            # восстановить pulse студента, опросив его practice_fn на узлах сетки
            for j in range(tg_b.shape[0]):
                pulse[j, b] = s["practice_fn"](tg_b[j])
            # item'ы студента сидят в целых временах 1..Ni -> индекс в общей сетке
            it = s["item_times"].round().long().clamp(0, T - 1)      # (Ni,)
            item_step[b, :Ni] = it
            q_vecs[b, :Ni] = s["q_vecs"]
            item_idx[b, :Ni] = s["item_idx"]
            y_true[b, :Ni] = s["y_true"]
            mask[b, :Ni] = True

        self.pulse = pulse
        self.item_step = item_step
        self.q_vecs = q_vecs
        self.item_idx = item_idx
        self.y_true = y_true
        self.mask = mask

    def to(self, device):
        for name in ("t_grid", "k0", "pulse", "item_step", "q_vecs",
                     "item_idx", "y_true", "mask"):
            setattr(self, name, getattr(self, name).to(device))
        return self


def make_batches(students: List[Dict[str, torch.Tensor]], K: int,
                 batch_size: int,
                 sort_by_length: bool = True) -> List[StudentBatch]:
    """Режет датасет на батчи. sort_by_length группирует похожие длины, чтобы
    минимизировать паддинг и лишние шаги интеграции."""
    order = list(range(len(students)))
    if sort_by_length:
        order.sort(key=lambda i: students[i]["item_times"].shape[0])
    batches = []
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batches.append(StudentBatch([students[i] for i in idx], K))
    return batches


# =============================================================================
# 3. Батчевая интеграция
# =============================================================================

def integrate_batch(model, batch: StudentBatch, integrator_step: float):
    """Возвращает траекторию (T, B, K) для всего батча одним odeint."""
    wrapper = BatchedODEWrapper(model.ode_func)
    pulse, t_grid = batch.pulse, batch.t_grid
    Tn = t_grid.shape[0]

    def practice_fn(t):
        # piecewise-constant: тот же searchsorted, что в лоадере
        j = torch.searchsorted(t_grid, t).clamp(0, Tn - 1)
        return pulse[j]                                              # (B, K)

    wrapper.set_practice(practice_fn)
    traj = odeint(wrapper, batch.k0, t_grid, method="rk4",
                  options={"step_size": integrator_step})
    return traj                                                     # (T, B, K)


def _gather_k_at_items(traj: torch.Tensor, batch: StudentBatch) -> torch.Tensor:
    """k в моменты item'ов: (B, L, K). item_times целые -> точное попадание на
    узлы сетки, эквивалентно линейной интерполяции _interp_traj в этих точках."""
    B, L = batch.item_step.shape
    bidx = torch.arange(B, device=traj.device).unsqueeze(1).expand(B, L)  # (B,L)
    return traj[batch.item_step, bidx]                              # (B, L, K)


# =============================================================================
# 4. Батчевый лосс (data-term + регуляризация B); ODE-residual опционален
# =============================================================================

def batched_loss(model, traj: torch.Tensor, batch: StudentBatch) -> dict:
    cfg = model.config
    k_at = _gather_k_at_items(traj, batch)                          # (B, L, K)
    m = batch.mask                                                  # (B, L)

    k_flat = k_at[m]                                                # (M, K)
    q_flat = batch.q_vecs[m]                                        # (M, K)
    idx_flat = batch.item_idx[m]                                    # (M,)
    y_flat = batch.y_true[m]                                        # (M,)

    p_pred = model.emission(k_flat, q_flat, idx_flat)              # (M,)
    bce = F.binary_cross_entropy(p_pred.clamp(1e-6, 1 - 1e-6),
                                 y_flat, reduction="mean")

    B = model.ode_func.B
    l1 = B.abs().sum()
    l2 = (B ** 2).sum()
    total = bce + cfg.lambda_l1 * l1 + cfg.lambda_l2 * l2
    return {"total": total, "bce": bce.detach(),
            "l1": l1.detach(), "l2": l2.detach(), "n": int(m.sum())}


# =============================================================================
# 5. Батчевые train / eval
# =============================================================================

def train_one_epoch_batched(model, batches: List[StudentBatch],
                            optimizer, device: str = "cpu") -> dict:
    model.train()
    agg = {"total": 0.0, "bce": 0.0, "n_batches": 0}
    for batch in batches:
        batch.to(device)
        traj = integrate_batch(model, batch, model.config.integrator_step)
        loss = batched_loss(model, traj, batch)
        optimizer.zero_grad()
        loss["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        agg["total"] += float(loss["total"].detach())
        agg["bce"] += float(loss["bce"])
        agg["n_batches"] += 1
    nb = max(agg["n_batches"], 1)
    agg["total"] /= nb
    agg["bce"] /= nb
    return agg


@torch.no_grad()
def evaluate_batched(model, batches: List[StudentBatch],
                     device: str = "cpu") -> dict:
    model.eval()
    ys, ps = [], []
    for batch in batches:
        batch.to(device)
        traj = integrate_batch(model, batch, model.config.integrator_step)
        k_at = _gather_k_at_items(traj, batch)
        m = batch.mask
        p = model.emission(k_at[m], batch.q_vecs[m], batch.item_idx[m])
        ys.append(batch.y_true[m].cpu().numpy())
        ps.append(p.cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return {"auc": float(roc_auc_score(y, p)),
            "acc": float(((p > 0.5) == y).mean()),
            "brier": float(brier_score_loss(y, p)),
            "n": int(len(y))}
