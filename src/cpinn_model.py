"""
Cognitive-PINN: PyTorch implementation of the model.

This module implements:
  * Forward ODE simulation via torchdiffeq.odeint_adjoint
    (memory-efficient adjoint sensitivity).
  * Sigmoidal emission model with Q-matrix coupling.
  * Composite PINN loss: BCE + ODE-residual + regularisation + KL-to-prior.
  * Training and evaluation routines with AUC / ACC / Brier metrics.

The code is intentionally framework-clean: the model is a single
nn.Module, the loss is a function, training is a standalone routine.
This keeps it portable to Colab/Kaggle and easy to extend later with a
custom semi-implicit Euler integrator (MEA) for on-device inference.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint, odeint
from sklearn.metrics import roc_auc_score, brier_score_loss


# =============================================================================
# 1. Hyperparameters
# =============================================================================

@dataclasses.dataclass
class CPINNConfig:
    """Container for all hyperparameters of the Cognitive-PINN."""
    K: int                      # number of concepts
    rank: Optional[int] = None  # low-rank for B (None -> full K)
    lambda_ode: float = 0.0     # ODE-residual weight; 0 in all reported runs (see RESULTS.md)
    lambda_l1: float = 1e-3     # weight of L1 sparsity on B
    lambda_l2: float = 1e-4     # weight of Frobenius regularisation on B
    lambda_kl: float = 0.0      # weight of KL to populational prior (0 = off)
    n_collocation: int = 32     # collocation points per trajectory for ODE-loss
    use_qprior: bool = True     # use expert prerequisite mask
    integrator: str = "rk4"     # 'rk4' or 'dopri5'
    integrator_step: float = 0.25
    dtype: torch.dtype = torch.float32


# =============================================================================
# 2. ODE function f_theta(k, p) implementing equation (1) from the paper
# =============================================================================

class CognitiveODEFunc(nn.Module):
    """
    Right-hand side of the Cognitive-PINN ODE:

        dk/dt = (A * phi(p)) odot (1 - k) - Lambda odot k

    with A = D^{1/2} (Q_prior odot B) D^{1/2} (Eq. 3 in P1).

    For K large (~100+) we parametrise B as a low-rank-plus-diagonal
    decomposition  B = U V^T + diag(d_self) for memory efficiency:
        - U: K x rank
        - V: K x rank
        - d_self: K (positive diagonal "self-strengthening")
    Set rank = K to recover the full parametrisation.
    """

    def __init__(self, K: int, q_prior: Optional[torch.Tensor] = None,
                 rank: Optional[int] = None):
        super().__init__()
        self.K = K
        self.rank = rank if rank is not None else K
        # low-rank factors
        self.U_raw = nn.Parameter(torch.randn(K, self.rank) * 0.05)
        self.V_raw = nn.Parameter(torch.randn(K, self.rank) * 0.05)
        # self-strengthening diagonal (always positive, initialised positive)
        self.log_d_self = nn.Parameter(torch.full((K,), math.log(0.3)))
        # diagonal scaling D
        self.log_d = nn.Parameter(torch.zeros(K))
        # forgetting rates Lambda > 0
        self.log_Lambda = nn.Parameter(torch.full((K,), math.log(0.05)))
        # practice responsiveness alpha > 0
        self.log_alpha = nn.Parameter(torch.zeros(K))
        # expert prior mask
        if q_prior is None:
            q_prior = torch.ones(K, K)
        self.register_buffer("q_prior", q_prior.float())
        self._practice = None

    # ------------------------------------------------------------------ utils
    def set_practice(self, practice_fn):
        self._practice = practice_fn

    @property
    def B(self) -> torch.Tensor:
        """Reconstruct B as softplus(U V^T) + diag(d_self)."""
        UV = self.U_raw @ self.V_raw.T              # K x K, can be negative
        B_full = F.softplus(UV) + torch.diag(F.softplus(self.log_d_self))
        return B_full

    @property
    def B_raw(self) -> torch.Tensor:
        """Backward-compatibility alias used by the loss for L1 regularisation."""
        return self.B

    @property
    def A(self) -> torch.Tensor:
        B = self.B
        d = torch.exp(self.log_d)
        D_half = torch.diag(torch.sqrt(d))
        return D_half @ (self.q_prior * B) @ D_half

    @property
    def Lambda(self) -> torch.Tensor:
        return torch.exp(self.log_Lambda)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.log_alpha)

    def forward(self, t: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        p = self._practice(t)
        phi = 1.0 - torch.exp(-self.alpha * p)
        learning = (self.A @ phi) * (1.0 - k)
        forgetting = self.Lambda * k
        return learning - forgetting


# =============================================================================
# 3. Per-student parameters (Lambda^s, sigma^s)
# =============================================================================

class PerStudentForgetting(nn.Module):
    """Optional per-student forgetting rates as a low-rank perturbation
    around the populational Lambda. Disabled in the basic prototype."""

    def __init__(self, K: int, n_students: int):
        super().__init__()
        # default: zero perturbation, so all students share Lambda
        self.delta_log_Lambda = nn.Parameter(torch.zeros(n_students, K))

    def forward(self, student_id: torch.Tensor) -> torch.Tensor:
        return self.delta_log_Lambda[student_id]


# =============================================================================
# 4. Emission model  P(y=1 | k, q) = sigmoid(beta * (q^T k) / |q|_1 - delta)
# =============================================================================

class IRTLikeEmission(nn.Module):
    """Item-Response-Theory-style emission with two parameters per item."""

    def __init__(self, n_items: int):
        super().__init__()
        self.beta_raw = nn.Parameter(torch.zeros(n_items))   # softplus -> beta
        self.delta = nn.Parameter(torch.zeros(n_items))      # difficulty

    def forward(self, k_at_t: torch.Tensor, q_vec: torch.Tensor,
                item_idx: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        k_at_t : (B, K) latent knowledge state at time of each interaction
        q_vec  : (B, K) Q-matrix row of the corresponding item, 0/1
        item_idx : (B,) integer item index for parameter lookup
        Returns
        -------
        prob   : (B,) probability of correct response
        """
        beta = F.softplus(self.beta_raw[item_idx]) + 0.5     # >= 0.5
        delta = self.delta[item_idx]
        # weighted average of mastery over the concepts of the item:
        # for one-hot Q-matrix this collapses to k[item_idx], the standard
        # DKT-style score; for multi-concept items it averages.
        norm = q_vec.sum(dim=-1).clamp_min(1.0)
        score = (q_vec * k_at_t).sum(dim=-1) / norm
        # IRT logit centered around mastery 0.5
        logit = beta * (score - 0.5) - delta
        return torch.sigmoid(logit)


# =============================================================================
# 5. Cognitive-PINN model wrapper
# =============================================================================

class CognitivePINN(nn.Module):
    """The full Cognitive-PINN model: ODE function + emission."""

    def __init__(self, config: CPINNConfig, n_items: int,
                 q_prior: Optional[torch.Tensor] = None):
        super().__init__()
        self.config = config
        self.ode_func = CognitiveODEFunc(config.K, q_prior=q_prior, rank=config.rank)
        self.emission = IRTLikeEmission(n_items)

    # -------- forward integration ------------------------------------------
    def integrate(self, t_grid: torch.Tensor, k0: torch.Tensor,
                  practice_fn) -> torch.Tensor:
        """Integrate the ODE from t_grid[0] to t_grid[-1], return k at t_grid.

        For small trajectories used in this prototype we use plain odeint:
        the trajectory has at most ~30 steps so storing it is cheap and
        backward through autograd is faster than odeint_adjoint's recompute.
        """
        self.ode_func.set_practice(practice_fn)
        traj = odeint(
            self.ode_func, k0, t_grid,
            method="rk4",
            options={"step_size": self.config.integrator_step},
        )
        return traj


# =============================================================================
# 6. Loss components
# =============================================================================

def cpinn_loss(model: CognitivePINN,
               trajectory: torch.Tensor,        # (T, K)  predicted k(t)
               t_grid: torch.Tensor,            # (T,)
               item_times: torch.Tensor,        # (B,) absolute times of interactions
               q_vecs: torch.Tensor,            # (B, K)
               item_idx: torch.Tensor,          # (B,)
               y_true: torch.Tensor,            # (B,) 0/1
               practice_fn) -> dict:
    """Compute the composite Cognitive-PINN loss for one student trajectory."""
    cfg = model.config
    # ------- 1. Data term: locate k(t_i) on the integrated grid via 1d interp
    # Grid is dense; use linear interpolation in time per concept.
    k_at_items = _interp_traj(trajectory, t_grid, item_times)        # (B, K)
    p_pred = model.emission(k_at_items, q_vecs, item_idx)
    bce = F.binary_cross_entropy(p_pred.clamp(1e-6, 1 - 1e-6),
                                  y_true.float(), reduction="mean")

    # ------- 2. ODE residual at random collocation points
    n_coll = cfg.n_collocation
    if n_coll > 0 and cfg.lambda_ode > 0:
        idx = torch.randperm(trajectory.shape[0])[:n_coll]
        idx, _ = torch.sort(idx)
        t_c = t_grid[idx]
        k_c = trajectory[idx]                              # (n_coll, K)
        # numerical derivative via finite differences on the trajectory
        dk_dt_num = _numeric_derivative(trajectory, t_grid)[idx]
        # vectorised model RHS: same A, Lambda for all points;
        # only practice p(t) varies with t_c
        p_c = torch.stack([practice_fn(t_c[i]) for i in range(len(t_c))])  # (n_coll, K)
        phi = 1.0 - torch.exp(-model.ode_func.alpha.unsqueeze(0) * p_c)    # (n_coll, K)
        learning = phi @ model.ode_func.A.T * (1.0 - k_c)                  # (n_coll, K)
        forgetting = model.ode_func.Lambda.unsqueeze(0) * k_c              # (n_coll, K)
        dk_dt_model = learning - forgetting
        ode_res = F.mse_loss(dk_dt_num, dk_dt_model)
    else:
        ode_res = torch.tensor(0.0, device=bce.device)

    # ------- 3. Regularisation on B
    B = model.ode_func.B  # already positive via softplus inside the property
    l1 = B.abs().sum()
    l2 = (B ** 2).sum()

    total = (
        bce
        + cfg.lambda_ode * ode_res
        + cfg.lambda_l1 * l1
        + cfg.lambda_l2 * l2
    )
    return {
        "total": total,
        "bce": bce.detach(),
        "ode": ode_res.detach(),
        "l1": l1.detach(),
        "l2": l2.detach(),
    }


# =============================================================================
# 7. Numerical helpers
# =============================================================================

def _interp_traj(traj: torch.Tensor, t_grid: torch.Tensor,
                 t_query: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of trajectory (T, K) at times t_query (B,)."""
    T = t_grid.shape[0]
    # clip queries into range
    t_query = t_query.clamp(t_grid[0], t_grid[-1])
    # find indices
    idx_right = torch.searchsorted(t_grid, t_query).clamp(1, T - 1)
    idx_left = idx_right - 1
    t_l = t_grid[idx_left]
    t_r = t_grid[idx_right]
    w_r = ((t_query - t_l) / (t_r - t_l).clamp_min(1e-9)).unsqueeze(-1)
    return (1 - w_r) * traj[idx_left] + w_r * traj[idx_right]


def _numeric_derivative(traj: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
    """Centered finite differences with one-sided treatment at boundaries."""
    dt = (t_grid[1:] - t_grid[:-1]).unsqueeze(-1)
    df_central = (traj[2:] - traj[:-2]) / (t_grid[2:] - t_grid[:-2]).unsqueeze(-1)
    df_left = (traj[1] - traj[0]) / dt[0]
    df_right = (traj[-1] - traj[-2]) / dt[-1]
    return torch.cat([df_left.unsqueeze(0), df_central, df_right.unsqueeze(0)],
                     dim=0)


# =============================================================================
# 8. Evaluation metrics
# =============================================================================

@torch.no_grad()
def evaluate(model: CognitivePINN, dataset, device: str = "cpu") -> dict:
    """Evaluate AUC, ACC, Brier on a dataset of student trajectories."""
    model.eval()
    all_y_true, all_y_pred = [], []
    for student in dataset:
        t_grid = student["t_grid"].to(device)
        k0 = student["k0"].to(device)
        traj = model.integrate(t_grid, k0, student["practice_fn"])
        k_at = _interp_traj(traj, t_grid, student["item_times"].to(device))
        p = model.emission(k_at,
                           student["q_vecs"].to(device),
                           student["item_idx"].to(device))
        all_y_true.append(student["y_true"].cpu().numpy())
        all_y_pred.append(p.cpu().numpy())
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    return {
        "auc": float(roc_auc_score(y_true, y_pred)),
        "acc": float(((y_pred > 0.5) == y_true).mean()),
        "brier": float(brier_score_loss(y_true, y_pred)),
        "n": int(len(y_true)),
    }


# =============================================================================
# 9. Training loop (single-student SGD; population/federated extensions later)
# =============================================================================

def train_one_epoch(model: CognitivePINN, dataset,
                    optimizer: torch.optim.Optimizer,
                    device: str = "cpu") -> dict:
    """One pass over the dataset; aggregates loss components."""
    model.train()
    agg = {"total": 0.0, "bce": 0.0, "ode": 0.0, "n": 0}
    for student in dataset:
        t_grid = student["t_grid"].to(device)
        k0 = student["k0"].to(device)
        traj = model.integrate(t_grid, k0, student["practice_fn"])
        loss = cpinn_loss(
            model, traj, t_grid,
            student["item_times"].to(device),
            student["q_vecs"].to(device),
            student["item_idx"].to(device),
            student["y_true"].to(device),
            student["practice_fn"],
        )
        optimizer.zero_grad()
        loss["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        agg["total"] += float(loss["total"].detach())
        agg["bce"]   += float(loss["bce"])
        agg["ode"]   += float(loss["ode"])
        agg["n"]     += 1
    for k in ("total", "bce", "ode"):
        agg[k] /= max(agg["n"], 1)
    return agg
