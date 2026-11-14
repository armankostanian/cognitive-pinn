"""
mac_common.py — общий код трёх контрольных экспериментов (MacBook, CPU/MPS).

Даёт: выбор устройства, загрузку Junyi fold 0, батчи, обучение с ФИКСИРОВАННЫМ
числом эпох (без early-stopping), извлечение A/Lambda, корреляции с контролем
конфаундера. Математика модели не изменяется: конфиг = боевой
(lambda_ode=0, lambda_l1=0, lambda_l2=0, integrator_step=0.25, rk4).
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cognitive-pinn", "src"))
for p in (SRC, os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

from cpinn_model import CPINNConfig, CognitivePINN           # noqa: E402
import cpinn_batched as cb                                    # noqa: E402
from cpinn_clean_loader import load_clean                     # noqa: E402

EPOCHS = 25
BATCH_SIZE = 512          # Mac unified memory; поднимите до 1024, если памяти хватает
INTEGRATOR_STEP = 0.25
LR = 1e-2                 # как в боевых прогонах run_cpinn_junyi_batched


def pick_device() -> str:
    """MPS (Apple GPU), если доступен и работает, иначе CPU."""
    if torch.backends.mps.is_available():
        try:  # smoke-test: базовые операции пайплайна
            x = torch.randn(64, 8, device="mps")
            _ = torch.searchsorted(torch.arange(10., device="mps"), x[:, 0].abs())
            _ = (x @ x.T).sum().item()
            return "mps"
        except Exception as e:  # noqa: BLE001
            print(f"[device] MPS есть, но smoke-test не прошёл ({e}); использую CPU")
    return "cpu"


def make_model(K: int, seed: int, device: str) -> CognitivePINN:
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = CPINNConfig(K=K, integrator="rk4", integrator_step=INTEGRATOR_STEP,
                      lambda_ode=0.0, lambda_l1=0.0, lambda_l2=0.0)
    model = CognitivePINN(cfg, n_items=K, q_prior=torch.ones(K, K))
    return model.to(device)


def load_junyi_fold0(dataset: str = "junyi", clean_root: str = "clean"):
    bundle = load_clean(dataset, fold=0, clean_root=clean_root)
    return bundle


def build_batches(students, K: int, device: str):
    t0 = time.time()
    batches = cb.make_batches(students, K, batch_size=BATCH_SIZE, sort_by_length=True)
    for b in batches:
        b.to(device)
    print(f"[batches] {len(students)} студентов -> {len(batches)} батчей "
          f"за {time.time()-t0:.1f} c (batch_size={BATCH_SIZE})")
    return batches


def train_fixed(model, train_batches, val_batches, device: str,
                epochs: int = EPOCHS, tag: str = "") -> dict:
    """Фиксированные epochs, БЕЗ early-stopping (как в 5-сидовом контроле)."""
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    hist = []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr = cb.train_one_epoch_batched(model, train_batches, opt, device=device)
        ev = cb.evaluate_batched(model, val_batches, device=device)
        dt = time.time() - t0
        hist.append({"ep": ep, "loss": tr["total"], "val_auc": ev["auc"], "sec": dt})
        print(f"[{tag}] ep {ep:02d}/{epochs}  loss={tr['total']:.4f}  "
              f"val_auc={ev['auc']:.4f}  {dt:.1f} c")
        if ep == 1:
            print(f"[{tag}] прогноз полного обучения: ~{dt*epochs/60:.1f} мин")
    return {"val_auc": hist[-1]["val_auc"], "history": hist}


def extract_params(model):
    A = model.ode_func.A.detach().cpu().numpy()
    Lam = model.ode_func.Lambda.detach().cpu().numpy()
    return A, Lam


def spearman_with_partial(Lam: np.ndarray, results_dir: str = "."):
    """rho(Lambda, difficulty) и частная rho при контроле числа наблюдений."""
    from scipy.stats import spearmanr, pearsonr, rankdata
    here = os.path.dirname(os.path.abspath(__file__))
    diff = np.load(os.path.join(here, "topic_difficulty.npy"))
    cnt = np.load(os.path.join(here, "topic_count.npy"))
    rho, p = spearmanr(Lam, diff)

    def resid(a, b):
        b1 = np.c_[np.ones_like(b), b]
        beta = np.linalg.lstsq(b1, a, rcond=None)[0]
        return a - b1 @ beta

    rx, ry, rz = rankdata(Lam), rankdata(diff), rankdata(cnt)
    pr, pp = pearsonr(resid(rx, rz), resid(ry, rz))
    return {"rho": float(rho), "p": float(p), "partial_rho": float(pr), "partial_p": float(pp)}


def edge_auc(A: np.ndarray, graph_path: str) -> float:
    from sklearn.metrics import roc_auc_score
    G = np.load(graph_path)
    K = A.shape[0]
    off = ~np.eye(K, dtype=bool)
    y = (G[off] > 0).astype(int)
    return float(roc_auc_score(y, np.abs(A[off])))
