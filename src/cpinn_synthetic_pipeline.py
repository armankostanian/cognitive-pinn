"""
Synthetic data generator for Cognitive-PINN end-to-end testing.

Creates a virtual cohort with known ground-truth parameters, samples
practice schedules and binary responses, then runs the training loop to
verify:
  (a) AUC on held-out interactions improves above chance,
  (b) recovered B / Lambda / alpha approach the ground truth.

Run:
    python cpinn_synthetic_pipeline.py
"""

from __future__ import annotations

import math
import os
import time
from typing import Callable, List, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from cpinn_model import (
    CPINNConfig, CognitivePINN,
    train_one_epoch, evaluate,
)

# -----------------------------------------------------------------------------
# 1. Synthetic course parameters
# -----------------------------------------------------------------------------

K = 5                 # five concepts
N_ITEMS = 20          # twenty items in the item bank
N_STUDENTS = 24       # smoke test cohort
T_HORIZON = 15.0      # days
SEED = 1234


def make_ground_truth(K: int) -> dict:
    """Designed-on-paper parameters for the synthetic course.

    A is sparse with a clear prerequisite chain 0 -> 1 -> 2 -> 3 -> 4
    plus a small lateral edge 1 -> 3.  Large diagonal => quick mastery.
    """
    A = np.zeros((K, K))
    for i in range(K - 1):
        A[i + 1, i] = 0.50                # prerequisite chain (strong)
    A[3, 1] = 0.20                        # lateral edge
    np.fill_diagonal(A, 0.60)             # strong self-strengthening
    Lam = np.array([0.04, 0.05, 0.06, 0.05, 0.07])[:K]
    alpha = np.array([1.0, 1.2, 0.9, 1.1, 1.0])[:K]
    return dict(A=A, Lam=Lam, alpha=alpha)


def make_q_prior(theta_star: dict, K: int) -> torch.Tensor:
    """Expert prior mask: ones where ground-truth A is non-zero."""
    return torch.tensor((theta_star["A"] > 0).astype(np.float32))


def make_q_matrix(n_items: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """Each item touches 1-2 concepts."""
    Q = np.zeros((n_items, K), dtype=np.float32)
    for i in range(n_items):
        n_concepts = rng.choice([1, 2], p=[0.7, 0.3])
        chosen = rng.choice(K, size=n_concepts, replace=False)
        Q[i, chosen] = 1.0
    return Q


# -----------------------------------------------------------------------------
# 2. Forward simulation of the ground-truth ODE for one student
# -----------------------------------------------------------------------------

def make_practice_fn(K: int, rng: np.random.Generator) -> Callable:
    """Random practice schedule: piecewise-constant intensity per concept."""
    n_blocks = 12
    block_len = T_HORIZON / n_blocks
    pattern = np.zeros((n_blocks, K))
    for b in range(n_blocks):
        # in each block one concept gets practised; sometimes none (rest)
        if rng.random() < 0.15:
            continue
        j = rng.integers(K)
        pattern[b, j] = rng.choice([0.5, 1.0, 1.5])

    def practice_np(t: float) -> np.ndarray:
        b = min(int(t / block_len), n_blocks - 1)
        return pattern[b]

    def practice_torch(t: torch.Tensor) -> torch.Tensor:
        # torch interface for the model
        b = min(int(float(t) / block_len), n_blocks - 1)
        return torch.tensor(pattern[b], dtype=torch.float32)

    return practice_np, practice_torch


def simulate_student_traj(theta_star: dict,
                          practice_np: Callable,
                          k0: np.ndarray,
                          t_eval: np.ndarray) -> np.ndarray:
    A = theta_star["A"]
    alpha = theta_star["alpha"]
    Lam = theta_star["Lam"]

    def rhs(t, k):
        p = practice_np(t)
        phi = 1 - np.exp(-alpha * p)
        return (A @ phi) * (1 - k) - Lam * k

    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), k0, t_eval=t_eval,
                    method="LSODA", rtol=1e-8, atol=1e-10)
    return sol.y  # (K, T)


# -----------------------------------------------------------------------------
# 3. Sample interactions from the latent trajectory
# -----------------------------------------------------------------------------

def emit_interactions(traj: np.ndarray, t_eval: np.ndarray,
                      Q: np.ndarray, n_interactions: int,
                      beta: float, rng: np.random.Generator,
                      delta_global: np.ndarray = None) -> dict:
    """Generate (item, time, response) tuples by sampling from the
    ground-truth IRT-like emission."""
    K, T = traj.shape
    t_idx = rng.choice(np.arange(T), size=n_interactions, replace=False)
    t_idx = np.sort(t_idx)
    item_times = t_eval[t_idx]
    item_idx = rng.integers(0, Q.shape[0], size=n_interactions)
    q_vecs = Q[item_idx]
    k_at = traj[:, t_idx].T
    norm = q_vecs.sum(axis=1).clip(1, None)
    score = (q_vecs * k_at).sum(axis=1) / norm
    if delta_global is None:
        delta_global = np.full(Q.shape[0], 0.3)  # mild fixed difficulty
    logit = beta * score - delta_global[item_idx]
    p_correct = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n_interactions) < p_correct).astype(np.int64)
    return dict(
        item_times=item_times,
        item_idx=item_idx,
        q_vecs=q_vecs,
        y_true=y,
    )


# -----------------------------------------------------------------------------
# 4. Build the dataset
# -----------------------------------------------------------------------------

def build_dataset(seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    theta_star = make_ground_truth(K)
    Q = make_q_matrix(N_ITEMS, K, rng)

    t_eval = np.linspace(0.0, T_HORIZON, 61)  # finer for sampling, RK4 still steps coarsely

    students = []
    for s in range(N_STUDENTS):
        practice_np, practice_torch = make_practice_fn(K, rng)
        k0 = rng.uniform(0.05, 0.2, size=K)
        traj = simulate_student_traj(theta_star, practice_np, k0, t_eval)
        interactions = emit_interactions(
            traj, t_eval, Q,
            n_interactions=36,
            beta=4.0, rng=rng,
        )
        # Convert to torch
        students.append({
            "t_grid": torch.tensor(t_eval, dtype=torch.float32),
            "k0": torch.tensor(k0, dtype=torch.float32),
            "practice_fn": practice_torch,
            "item_times": torch.tensor(interactions["item_times"], dtype=torch.float32),
            "item_idx": torch.tensor(interactions["item_idx"], dtype=torch.long),
            "q_vecs": torch.tensor(interactions["q_vecs"], dtype=torch.float32),
            "y_true": torch.tensor(interactions["y_true"], dtype=torch.float32),
        })

    return dict(
        theta_star=theta_star,
        Q=Q,
        students=students,
        t_eval=t_eval,
    )


# -----------------------------------------------------------------------------
# 5. Recover learned A from the trained model and compare with ground truth
# -----------------------------------------------------------------------------

def compare_A(model_A: np.ndarray, true_A: np.ndarray) -> dict:
    """Compute structural recovery diagnostics between learned and true A."""
    rel_err = np.linalg.norm(model_A - true_A) / np.linalg.norm(true_A)
    # Edge-recovery via thresholding
    learned_edges = model_A > 0.05
    true_edges = true_A > 0.05
    tp = int((learned_edges & true_edges).sum())
    fp = int((learned_edges & ~true_edges).sum())
    fn = int((~learned_edges & true_edges).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return dict(rel_err=rel_err, precision=precision, recall=recall, f1=f1)


# -----------------------------------------------------------------------------
# 6. Train / evaluate end-to-end
# -----------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"[1/5] Building synthetic dataset: K={K}, students={N_STUDENTS}")
    data = build_dataset()
    theta_star = data["theta_star"]

    # 80/20 train/val split per student's interactions
    train_students, val_students = [], []
    rng = np.random.default_rng(SEED)
    for s in data["students"]:
        n = s["y_true"].shape[0]
        perm = rng.permutation(n)
        cut = int(0.8 * n)
        idx_tr = torch.tensor(np.sort(perm[:cut]))
        idx_va = torch.tensor(np.sort(perm[cut:]))
        train_students.append({
            **s,
            "item_times": s["item_times"][idx_tr],
            "item_idx":   s["item_idx"][idx_tr],
            "q_vecs":     s["q_vecs"][idx_tr],
            "y_true":     s["y_true"][idx_tr],
        })
        val_students.append({
            **s,
            "item_times": s["item_times"][idx_va],
            "item_idx":   s["item_idx"][idx_va],
            "q_vecs":     s["q_vecs"][idx_va],
            "y_true":     s["y_true"][idx_va],
        })

    print(f"[2/5] Building Cognitive-PINN: K={K}, n_items={N_ITEMS}")
    cfg = CPINNConfig(K=K, lambda_ode=0.1, lambda_l1=5e-3, lambda_l2=1e-3,
                      n_collocation=12, integrator="rk4",
                      integrator_step=0.25)
    q_prior = make_q_prior(theta_star, K)
    model = CognitivePINN(cfg, n_items=N_ITEMS, q_prior=q_prior)

    optim = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)

    print(f"[3/5] Training. Initial Lambda = {model.ode_func.Lambda.detach().numpy()}")
    n_epochs = 20
    history = []
    best_auc = 0.0
    best_state = None
    for ep in range(1, n_epochs + 1):
        t0 = time.time()
        log_tr = train_one_epoch(model, train_students, optim)
        elapsed = time.time() - t0
        eval_va = evaluate(model, val_students)
        history.append({"epoch": ep, **log_tr, **eval_va})
        if eval_va["auc"] > best_auc:
            best_auc = eval_va["auc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:3d}  loss={log_tr['total']:.4f}  "
                  f"bce={log_tr['bce']:.4f}  ode={log_tr['ode']:.4f}  "
                  f"val_auc={eval_va['auc']:.3f}  ({elapsed:.1f}s)")

    # restore best checkpoint for diagnostics
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  best val_auc = {best_auc:.3f}")

    # ---------- Recovery diagnostics ----------
    print(f"[4/5] Recovery diagnostics")
    A_learned = model.ode_func.A.detach().numpy()
    Lam_learned = model.ode_func.Lambda.detach().numpy()
    alpha_learned = model.ode_func.alpha.detach().numpy()

    diag = compare_A(A_learned, theta_star["A"])
    print(f"  A: rel_err={diag['rel_err']:.3f}  "
          f"precision={diag['precision']:.2f}  recall={diag['recall']:.2f}  "
          f"f1={diag['f1']:.2f}")
    print(f"  Lambda  true={theta_star['Lam']}")
    print(f"  Lambda learn={Lam_learned}")
    print(f"  alpha   true={theta_star['alpha']}")
    print(f"  alpha  learn={alpha_learned}")

    final = history[-1]
    print(f"  final val: auc={final['auc']:.3f} acc={final['acc']:.3f} brier={final['brier']:.3f}")

    # ---------- Plot ----------
    print(f"[5/5] Plotting training curves")
    os.makedirs("artifacts", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    eps = [h["epoch"] for h in history]
    axes[0].plot(eps, [h["total"] for h in history], label="total")
    axes[0].plot(eps, [h["bce"]   for h in history], label="bce")
    axes[0].plot(eps, [h["ode"]   for h in history], label="ode-residual")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend()
    axes[0].set_title("(a) Training loss components"); axes[0].grid(alpha=0.3)

    axes[1].plot(eps, [h["auc"] for h in history], color="C2", linewidth=2)
    axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation AUC")
    axes[1].set_title("(b) Held-out AUC"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0.45, 1.0)

    im_args = dict(cmap="Blues", vmin=0.0, vmax=max(theta_star["A"].max(), A_learned.max()))
    ax = axes[2]
    im = ax.imshow(A_learned, **im_args)
    ax.set_title("(c) Learned A (heatmap)")
    ax.set_xlabel("source concept"); ax.set_ylabel("target concept")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig("artifacts/cpinn_synthetic_training.png", dpi=140)
    print("  -> artifacts/cpinn_synthetic_training.png")

    fig2, axes2 = plt.subplots(1, 2, figsize=(9, 4))
    axes2[0].imshow(theta_star["A"], **im_args); axes2[0].set_title("Ground-truth A")
    axes2[1].imshow(A_learned, **im_args); axes2[1].set_title("Learned A")
    fig2.tight_layout()
    fig2.savefig("artifacts/cpinn_A_compare.png", dpi=140)
    print("  -> artifacts/cpinn_A_compare.png")

    return history, diag


if __name__ == "__main__":
    main()
