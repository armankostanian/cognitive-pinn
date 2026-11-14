"""
exactkt.py — оценивание модели динамики знаний по точным траекториям.

Модель:     dk_i/dt = u_i (1 - k_i) - Lam_i k_i,   u = A phi(p),  phi_j(p) = 1 - exp(-alpha_j p)
Наблюдение: y ~ Binomial(n_items, sigmoid(beta_i k_i + gamma_i + eta_{i,form})),  eta_{i,F01} = 0
Ученик n:   z_n = (logit k0_n1..K, b_n1..K),   Lam_{n,i} = Lam_i * exp(b_{n,i})
Оценивание: Laplace-EM — совместная мода по (глобальные параметры, z_n) при гауссовском
            априорном z_n ~ N(m, diag S); затем обновление (m, S) с учётом апостериорной
            дисперсии из блоков гессиана.
"""
from __future__ import annotations
import math
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
HS = 1.0 / 24.0
ROT = ["F01","F02","F03","F04","F05","F06","F07","F08","F09","F10","F11","F12",
       "F01","F02","F03","F04","F05","F06","F07","F08"]
FORMS = [f"F{q:02d}" for q in range(1, 13)]


def registered_design(order="S1S2"):
    """20 замеров (P00-P19), интенсивности 10,4,4,10. Возвращает времена, формы,
    вход на каждом промежутке между соседними замерами (K=2)."""
    t = [0.0]
    for d in (7, 10, 14, 17): t += [d, d + HS]
    t.append(24.0)
    for d in (35, 38, 42, 45): t += [d, d + HS]
    t += [49.0, 63.0]
    t = np.array(t)
    P = np.zeros((len(t) - 1, 2))
    blocks = (0, 1) if order == "S1S2" else (1, 0)
    for b, pre_idx in zip(blocks, ([1, 3, 5, 7], [10, 12, 14, 16])):
        for n, q in zip(pre_idx, (10, 4, 4, 10)):
            P[n, b] = q
    form_idx = np.array([FORMS.index(f) for f in ROT])
    mid = np.array([9, 18])                       # P09, P18
    return dict(t=t, P=P, form=form_idx, mid=mid)


class Params(torch.nn.Module):
    def __init__(self, K=2, F=12, N=100, random_lambda=True):
        super().__init__()
        sp_inv = lambda x: math.log(math.expm1(x))
        self.K, self.F, self.N, self.rl = K, F, N, random_lambda
        A0 = torch.full((K, K), sp_inv(0.5)); A0.fill_diagonal_(sp_inv(3.0))
        self.A_raw = torch.nn.Parameter(A0)
        self.alpha_raw = torch.nn.Parameter(torch.full((K,), sp_inv(0.1)))
        self.logLam = torch.nn.Parameter(torch.full((K,), math.log(0.05)))
        self.beta_raw = torch.nn.Parameter(torch.full((K,), sp_inv(4.0)))
        self.gamma = torch.nn.Parameter(torch.full((K,), -1.5))
        self.eta = torch.nn.Parameter(torch.zeros(K, F - 1))           # формы F02..F12
        self.z = torch.nn.Parameter(torch.zeros(N, 2 * K))               # logit k0, b
        # гиперпараметры априорного распределения z (обновляются в M-шаге EM)
        self.m = torch.zeros(2 * K)
        self.S = torch.cat([torch.full((K,), 1.0), torch.full((K,), 0.1 if random_lambda else 1e-8)])

    def glob(self):
        sp = torch.nn.functional.softplus
        return dict(A=sp(self.A_raw), alpha=sp(self.alpha_raw), Lam=torch.exp(self.logLam),
                    beta=sp(self.beta_raw), gamma=self.gamma,
                    eta=torch.cat([torch.zeros(self.K, 1), self.eta], 1))


def trajectory(g, z, design):
    """Уровни освоения во всех замерах: точные шаги, векторизовано по ученикам. (N, M, K)"""
    K = g["A"].shape[0]
    k = torch.sigmoid(z[:, :K])
    Lam_n = g["Lam"][None, :] * torch.exp(z[:, K:])
    t = torch.as_tensor(design["t"]); P = torch.as_tensor(design["P"])
    out = [k]
    for m in range(len(t) - 1):
        h = t[m + 1] - t[m]
        if P.dim() == 2:
            u = (g["A"] @ (1 - torch.exp(-g["alpha"] * P[m])))[None, :]
        else:
            u = (1 - torch.exp(-g["alpha"] * P[:, m])) @ g["A"].T          # (N,K)
        s = u + Lam_n
        ks = u / s
        k = ks + (k - ks) * torch.exp(-s * h)
        out.append(k)
    return torch.stack(out, 1)


def loglik_per_learner(model, Y, NI, mask, design):
    g = model.glob()
    k = trajectory(g, model.z, design)                                   # (N,M,K)
    eta = g["eta"][:, torch.as_tensor(design["form"])].T                 # (M,K)
    w = g["beta"] * k + g["gamma"] + eta[None]
    ll = Y * torch.nn.functional.logsigmoid(w) + (NI - Y) * torch.nn.functional.logsigmoid(-w)
    return (ll * mask).sum((1, 2))                                        # (N,)


def objective(model, Y, NI, mask, design):
    ll = loglik_per_learner(model, Y, NI, mask, design)
    extra = 0.0
    if getattr(model, 'beta_prior', None) is not None:
        mu, sd = model.beta_prior
        lb = torch.log(torch.nn.functional.softplus(model.beta_raw))
        extra = 0.5 * (((lb - math.log(mu)) / sd) ** 2).sum()
    z = model.z; m, S = model.m, model.S
    lp = -0.5 * (((z - m) ** 2) / S).sum(1)
    if not model.rl:
        lp = -0.5 * (((z[:, :model.K] - m[:model.K]) ** 2) / S[:model.K]).sum(1)
    return -(ll + lp).sum() + extra


def fit(Y, NI, mask, design, random_lambda=True, em_iters=8, verbose=False,
        beta_prior=None, fix_beta=None, s_floor=0.05):
    N = Y.shape[0]
    model = Params(N=N, random_lambda=random_lambda)
    model.beta_prior = beta_prior
    if fix_beta is not None:
        sp_inv = lambda x: math.log(math.expm1(x))
        model.beta_raw.data = torch.tensor([sp_inv(b) for b in fix_beta])
        model.beta_raw.requires_grad_(False)
    if not random_lambda:
        model.z.data[:, model.K:] = 0.0
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    for it in range(em_iters):
        opt = torch.optim.LBFGS(params, lr=1, max_iter=400, tolerance_grad=1e-8,
                                tolerance_change=1e-12, history_size=50, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            J = objective(model, Y, NI, mask, design)
            J.backward()
            if not random_lambda: model.z.grad[:, model.K:] = 0.0
            return J
        opt.step(closure)
        # E/M-шаг: апостериорные дисперсии из блоков гессиана по z_n
        J = objective(model, Y, NI, mask, design)
        gz = torch.autograd.grad(J, model.z, create_graph=True)[0]
        D = 2 * model.K
        Hb = torch.zeros(N, D, D)
        for d in range(D):
            v = torch.zeros_like(model.z); v[:, d] = 1.0
            Hb[:, :, d] = torch.autograd.grad(gz, model.z, grad_outputs=v, retain_graph=True)[0]
        dims = D if random_lambda else model.K
        V = torch.linalg.inv(Hb[:, :dims, :dims] + 1e-10 * torch.eye(dims))
        zh = model.z.detach()[:, :dims]
        mnew = zh.mean(0)
        if random_lambda: mnew[model.K:] = 0.0                             # среднее b = 0
        Snew = ((zh - mnew) ** 2 + torch.diagonal(V, dim1=1, dim2=2)).mean(0)
        dS = (Snew - model.S[:dims]).abs().max().item()
        model.m[:dims] = mnew; Snew[:model.K] = Snew[:model.K].clamp_min(s_floor); model.S[:dims] = Snew.clamp_min(1e-6)
        if verbose:
            print(f"  EM {it}: J={J.item():.2f}  S={np.round(model.S.numpy(), 4)}")
        if dS < 1e-4: break
    return model


# ---------------------------- моделирование данных ----------------------------
TRUTH = dict(A=np.array([[5.0, 0.8], [1.5, 4.0]]), alpha=np.array([0.12, 0.10]),
             Lam=np.array([0.05, 0.07]), beta=np.array([5.0, 4.0]), gamma=np.array([-2.0, -1.5]))


def simulate(N, design, rng, tau=0.2, tau_A=0.0, reduced_frac=0.2, n_items=5, eta_sd=0.3, eta=None):
    K = 2
    if eta is None:
        eta = np.c_[np.zeros(K), rng.normal(0, eta_sd, (K, 11))]
    k = 1 / (1 + np.exp(-rng.normal(np.log(0.45 / 0.55), 0.6, (N, K))))
    Lam = TRUTH["Lam"][None] * np.exp(rng.normal(0, tau, (N, K)))
    Ad = TRUTH["A"][None] * np.exp(rng.normal(0, tau_A, (N, K, K)))
    t = design["t"]
    Pa, Pb = registered_design("S1S2")["P"], registered_design("S2S1")["P"]
    order = rng.random(N) < 0.5
    P = np.where(order[:, None, None], Pa[None], Pb[None])                 # (N,M-1,K)
    design["P_learner"] = P
    ks = [k]
    for m in range(len(t) - 1):
        u = np.einsum("nij,nj->ni", Ad, 1 - np.exp(-TRUTH["alpha"] * P[:, m]))
        s = u + Lam; kst = u / s
        k = kst + (k - kst) * np.exp(-s * (t[m + 1] - t[m])); ks.append(k)
    k = np.stack(ks, 1)
    w = TRUTH["beta"] * k + TRUTH["gamma"] + eta[:, design["form"]].T[None]
    Y = rng.binomial(n_items, 1 / (1 + np.exp(-w))).astype(float)
    mask = np.ones_like(Y)
    red = rng.random(N) < reduced_frac
    mask[np.ix_(red, design["mid"], np.arange(K))] = 0.0
    NI = np.full_like(Y, n_items)
    return (torch.tensor(Y), torch.tensor(NI), torch.tensor(mask)), eta


# ============================================================================
#   Маргинальное правдоподобие: адаптивные квадратуры Гаусса–Эрмита (AGHQ)
# ============================================================================
import itertools

class MParams(torch.nn.Module):
    """Глобальные параметры + гиперпараметры случайных эффектов (без z учеников)."""
    def __init__(self, K=2, F=12, random_lambda=True):
        super().__init__()
        sp_inv = lambda x: math.log(math.expm1(x))
        self.K, self.F, self.rl = K, F, random_lambda
        A0 = torch.full((K, K), sp_inv(0.5)); A0.fill_diagonal_(sp_inv(3.0))
        self.A_raw = torch.nn.Parameter(A0)
        self.alpha_raw = torch.nn.Parameter(torch.full((K,), sp_inv(0.1)))
        self.logLam = torch.nn.Parameter(torch.full((K,), math.log(0.05)))
        self.beta_raw = torch.nn.Parameter(torch.full((K,), sp_inv(4.0)))
        self.gamma = torch.nn.Parameter(torch.full((K,), -1.5))
        self.eta = torch.nn.Parameter(torch.zeros(K, F - 1))
        self.m0 = torch.nn.Parameter(torch.zeros(K))                     # среднее logit k0
        self.logS0 = torch.nn.Parameter(torch.zeros(K))                  # log дисперсии logit k0
        self.logT = torch.nn.Parameter(torch.full((K,), math.log(0.04)))  # log дисперсии b
        self.beta_prior = None
    glob = Params.glob
    def D(self): return 2 * self.K if self.rl else self.K


def neg_joint(mp, z, Y, NI, mask, design):
    """f_n(z) = -[log p(y_n|z) + log N(z; m, S)] для каждого ученика (и узла). z: (..., D)."""
    K = mp.K; sh = z.shape[:-1]; zf = z.reshape(-1, z.shape[-1])
    if not mp.rl:
        zf = torch.cat([zf, torch.zeros(zf.shape[0], K)], 1)
    g = mp.glob()
    rep = zf.shape[0] // Y.shape[0]
    dz = design
    if np.ndim(design["P"]) == 3 and rep > 1:
        dz = dict(design); dz["P"] = torch.as_tensor(design["P"]).repeat_interleave(rep, 0)
    k = trajectory(g, zf, dz)
    eta = g["eta"][:, torch.as_tensor(design["form"])].T
    w = g["beta"] * k + g["gamma"] + eta[None]
    Yr, NIr, Mr = (x.repeat_interleave(rep, 0) if rep > 1 else x for x in (Y, NI, mask))
    ll = (Yr * torch.nn.functional.logsigmoid(w) + (NIr - Yr) * torch.nn.functional.logsigmoid(-w)) * Mr
    ll = ll.sum((1, 2))
    S0 = torch.exp(mp.logS0); lp = -0.5 * (((zf[:, :K] - mp.m0) ** 2) / S0).sum(1) - 0.5 * mp.logS0.sum() \
         - 0.5 * K * math.log(2 * math.pi)
    if mp.rl:
        T = torch.exp(mp.logT); lp = lp - 0.5 * ((zf[:, K:] ** 2) / T).sum(1) - 0.5 * mp.logT.sum() \
             - 0.5 * K * math.log(2 * math.pi)
    return -(ll + lp).reshape(sh)


def inner_modes(mp, Y, NI, mask, design, z0=None, iters=25):
    """Мода и гессиан f_n по z_n для каждого ученика (Ньютон, векторизовано)."""
    N, D = Y.shape[0], mp.D()
    z = (torch.zeros(N, D) if z0 is None else z0.clone()).requires_grad_(True)
    with torch.no_grad():
        for p in mp.parameters(): p.requires_grad_(False)
    for it in range(iters):
        f = neg_joint(mp, z, Y, NI, mask, design)
        G = torch.autograd.grad(f.sum(), z, create_graph=True)[0]
        H = torch.zeros(N, D, D)
        for d in range(D):
            v = torch.zeros_like(z); v[:, d] = 1.0
            H[:, :, d] = torch.autograd.grad(G, z, grad_outputs=v, retain_graph=True)[0]
        H = 0.5 * (H + H.transpose(1, 2))
        H = torch.nan_to_num(H, nan=0.0, posinf=1e6, neginf=-1e6) + 1e-8 * torch.eye(D)
        try:
            ev, U = torch.linalg.eigh(H)
        except Exception:                      # плохо обусловленный блок: диагональная замена
            ev, U = torch.diagonal(H, dim1=1, dim2=2), torch.eye(D).expand(N, D, D).clone()
        ev = ev.clamp(min=1e-6, max=1e8)
        Hpd = U @ torch.diag_embed(ev) @ U.transpose(1, 2)
        step = torch.linalg.solve(Hpd, G.detach().unsqueeze(-1)).squeeze(-1)
        step = torch.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)
        nrm = step.norm(dim=1, keepdim=True)
        step = step * torch.clamp(2.0 / (nrm + 1e-30), max=1.0)        # ограничение длины шага
        with torch.no_grad():
            f0 = f.detach(); t = torch.ones(N, 1)
            for _ in range(12):                                  # покомпонентный backtracking
                fn = neg_joint(mp, z - t * step, Y, NI, mask, design)
                bad = (~(fn <= f0 + 1e-10)).unsqueeze(-1)       # NaN/inf считаются плохим шагом
                t = torch.where(bad, t * 0.5, t)
                if not bad.any(): break
            t = torch.where(bad, torch.zeros_like(t), t)         # нет улучшения — остаёмся на месте
            z = (z - t * step).detach().requires_grad_(True)
        if step.abs().max() < 1e-8: break
    for p in mp.parameters(): p.requires_grad_(True)
    return z.detach(), Hpd.detach()


def gh_nodes(D, n=3):
    x, w = np.polynomial.hermite.hermgauss(n)
    X = np.array(list(itertools.product(x, repeat=D))); W = np.prod(list(itertools.product(w, repeat=D)), 1)
    return torch.tensor(X), torch.tensor(W)


def marginal_nll(mp, Y, NI, mask, design, zh, H, X, W):
    """-sum_n log int exp(-f_n(z)) dz по AGHQ с узлами вокруг моды zh (фиксированы)."""
    L = torch.linalg.cholesky(torch.linalg.inv(H))                   # (N,D,D)
    Zq = zh[:, None, :] + math.sqrt(2) * torch.einsum("nij,qj->nqi", L, X)   # (N,Q,D)
    f = neg_joint(mp, Zq, Y, NI, mask, design)                         # (N,Q)
    f = torch.nan_to_num(f, nan=1e30, posinf=1e30)
    logdet = torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1) + 0.5 * X.shape[1] * math.log(2)
    a = -f + (X ** 2).sum(1)[None] + torch.log(W)[None]
    return -(torch.logsumexp(a, 1) + logdet).sum()


def fit_marginal(Y, NI, mask, design, random_lambda=True, fix_beta=None, beta_prior=None,
                 alpha_prior=(0.1, 1.0), outer=6, n_gh=3, verbose=False):
    mp = MParams(random_lambda=random_lambda)
    if fix_beta is not None:
        sp_inv = lambda x: math.log(math.expm1(x))
        mp.beta_raw.data = torch.tensor([sp_inv(b) for b in fix_beta]); mp.beta_raw.requires_grad_(False)
    X, W = gh_nodes(mp.D(), n_gh); zh = None
    for it in range(outer):
        zh, H = inner_modes(mp, Y, NI, mask, design, z0=zh)
        if fix_beta is not None: mp.beta_raw.requires_grad_(False)
        params = [p for p in mp.parameters() if p.requires_grad]
        opt = torch.optim.LBFGS(params, lr=1, max_iter=200, tolerance_grad=1e-7, tolerance_change=1e-11,
                                history_size=50, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            J = marginal_nll(mp, Y, NI, mask, design, zh, H, X, W)
            if beta_prior is not None:
                mu, sd = beta_prior
                J = J + 0.5 * (((torch.log(torch.nn.functional.softplus(mp.beta_raw)) - math.log(mu)) / sd) ** 2).sum()
            if alpha_prior is not None:
                mu, sd = alpha_prior
                J = J + 0.5 * (((torch.log(torch.nn.functional.softplus(mp.alpha_raw)) - math.log(mu)) / sd) ** 2).sum()
            J.backward(); return J
        J = opt.step(closure)
        if verbose:
            g = {k: v.detach().numpy() for k, v in mp.glob().items()}
            print(f"  outer {it}: NLL={J.item():.2f} Lam={np.round(g['Lam'],4)} alpha={np.round(g['alpha'],3)} "
                  f"beta={np.round(g['beta'],2)} tau={np.round(np.sqrt(np.exp(mp.logT.detach().numpy())),3)}")
    return mp
