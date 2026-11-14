"""
Параллельный префиксный scan для аффинной рекурсии k_{n+1} = m_n k_n + c_n (Лемма 1, Следствие 2)
и сравнение с последовательным точным шагом и с RK4.
Hillis–Steele: O(T log T) работы, O(log T) последовательных шагов.
"""
import time, math
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


def affine_scan(m, c):
    """m, c: (..., T). Возвращает (M, C) такие, что k_t = M_t * k_0 + C_t."""
    M, C = m.clone(), c.clone(); T = m.shape[-1]; d = 1
    while d < T:
        Mn, Cn = M.clone(), C.clone()
        Mn[..., d:] = M[..., d:] * M[..., :-d]
        Cn[..., d:] = M[..., d:] * C[..., :-d] + C[..., d:]
        M, C = Mn, Cn; d *= 2
    return M, C


def make_problem(B, K, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.rand(K, K, generator=g) * 2 + torch.eye(K) * 3
    Lam = torch.rand(K, generator=g) * 0.3 + 0.02
    alpha = torch.rand(K, generator=g) * 0.2 + 0.05
    P = torch.zeros(T, K); idx = torch.randint(0, K, (T,), generator=g)
    prac = torch.rand(T, generator=g) < 0.5
    P[torch.arange(T)[prac], idx[prac]] = torch.tensor([4.0, 10.0])[torch.randint(0, 2, (int(prac.sum()),), generator=g)]
    h = torch.where(prac, torch.full((T,), 1 / 24), torch.tensor([1.0, 3.0, 14.0])[torch.randint(0, 3, (T,), generator=g)])
    k0 = torch.rand(B, K, generator=g) * 0.6 + 0.2
    u = (A @ (1 - torch.exp(-alpha[:, None] * P.T))).T            # (T,K)
    return k0, u, Lam, h


def sequential_exact(k0, u, Lam, h):
    k = k0; out = []
    for t in range(u.shape[0]):
        s = u[t] + Lam; ks = u[t] / s
        k = ks + (k - ks) * torch.exp(-s * h[t]); out.append(k)
    return torch.stack(out, -1)                                     # (B,K,T)


def scan_exact(k0, u, Lam, h):
    s = u + Lam; m = torch.exp(-s * h[:, None]); c = (u / s) * (1 - m)   # (T,K)
    M, C = affine_scan(m.T, c.T)                                     # (K,T)
    return M[None] * k0[:, :, None] + C[None]                        # (B,K,T)


def rk4(k0, u, Lam, h, sub=4):
    k = k0; out = []
    for t in range(u.shape[0]):
        n = max(sub, int(math.ceil(h[t].item() / 0.25))); d = h[t] / n
        f = lambda x: u[t] * (1 - x) - Lam * x
        for _ in range(n):
            a1 = f(k); a2 = f(k + d / 2 * a1); a3 = f(k + d / 2 * a2); a4 = f(k + d * a3)
            k = k + d / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        out.append(k)
    return torch.stack(out, -1)


if __name__ == "__main__":
    B, K = 64, 5
    print(f"B={B} учеников, K={K} тем; интервалы: занятия 1 ч и паузы 1–14 сут")
    print(f"{'T':>7} | {'последоват.':>11} | {'scan':>9} | {'RK4 (h<=0.25)':>13} | {'|scan-посл|':>11} | {'|RK4-посл|':>10}")
    for T in (100, 1000, 10000):
        k0, u, Lam, h = make_problem(B, K, T)
        t0 = time.perf_counter(); a = sequential_exact(k0, u, Lam, h); t1 = time.perf_counter()
        b = scan_exact(k0, u, Lam, h); t2 = time.perf_counter()
        if T <= 1000:
            r = rk4(k0, u, Lam, h); t3 = time.perf_counter(); rt = f"{t3 - t2:10.3f} с"; re = f"{(r - a).abs().max():.1e}"
        else:
            rt, re = "   (пропуск)", "—"
        print(f"{T:>7} | {t1 - t0:9.3f} с | {t2 - t1:7.3f} с | {rt:>13} | {(b - a).abs().max():>11.1e} | {re:>10}")
