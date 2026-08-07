"""Sanity check: plug ground-truth theta into the model and measure AUC.
This bounds from above what any learner can possibly get on these data."""

import numpy as np
import torch
from cpinn_synthetic_pipeline import build_dataset, K, N_ITEMS, make_q_prior
from cpinn_model import CPINNConfig, CognitivePINN, evaluate
import torch.nn.functional as F

data = build_dataset()
theta = data["theta_star"]

cfg = CPINNConfig(K=K, integrator_step=0.25)
q_prior = make_q_prior(theta, K)
model = CognitivePINN(cfg, n_items=N_ITEMS, q_prior=q_prior)

# Plug ground-truth parameters: invert softplus and the D=I parametrisation
# so that A_inv = Q_prior * B_true = A_true (with D=I)
A_target = torch.tensor(theta["A"], dtype=torch.float32)
# we set B such that softplus(B_raw) * Q_prior = A_target
B_target = A_target / (q_prior + 1e-9)
# inverse softplus: log(exp(x) - 1)
B_raw_target = torch.log(torch.expm1(B_target.clamp_min(1e-6)))
B_raw_target = torch.where(q_prior > 0, B_raw_target, torch.full_like(B_raw_target, -10.0))

with torch.no_grad():
    model.ode_func.B_raw.copy_(B_raw_target)
    model.ode_func.log_d.zero_()           # D = I
    model.ode_func.log_Lambda.copy_(torch.log(torch.tensor(theta["Lam"], dtype=torch.float32)))
    model.ode_func.log_alpha.copy_(torch.log(torch.tensor(theta["alpha"], dtype=torch.float32)))

# Verify A reconstruction
A_reconstructed = model.ode_func.A.detach().numpy()
print("A_target:")
print(theta["A"])
print("A_reconstructed:")
print(A_reconstructed)
print("max abs diff:", np.max(np.abs(A_reconstructed - theta["A"])))

# Now train ONLY the IRT emission for a few epochs (item parameters are unknown)
optim = torch.optim.Adam(
    list(model.emission.parameters()), lr=5e-2,
)

# split
rng = np.random.default_rng(123)
train_students, val_students = [], []
for s in data["students"]:
    n = s["y_true"].shape[0]
    perm = rng.permutation(n)
    cut = int(0.8 * n)
    idx_tr = torch.tensor(np.sort(perm[:cut]))
    idx_va = torch.tensor(np.sort(perm[cut:]))
    train_students.append({**s, **{k: s[k][idx_tr] for k in ("item_times","item_idx","q_vecs","y_true")}})
    val_students.append({**s, **{k: s[k][idx_va] for k in ("item_times","item_idx","q_vecs","y_true")}})

from cpinn_model import cpinn_loss, train_one_epoch
# fix ODE params: only emission updated
for p in model.ode_func.parameters():
    p.requires_grad = False

for ep in range(15):
    model.train()
    for s in train_students:
        traj = model.integrate(s["t_grid"], s["k0"], s["practice_fn"])
        loss = cpinn_loss(model, traj, s["t_grid"], s["item_times"], s["q_vecs"],
                          s["item_idx"], s["y_true"], s["practice_fn"])
        optim.zero_grad()
        loss["total"].backward()
        optim.step()
    eva = evaluate(model, val_students)
    print(f"  ep {ep+1:2d}: val_auc={eva['auc']:.3f}  acc={eva['acc']:.3f}")
