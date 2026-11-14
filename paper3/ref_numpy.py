"""Эталон на NumPy (проверен): сегментированный scan и сопряжённая рекурсия."""
import numpy as np

def seg_scan(m, c, f):
    """m, c: (T, K); f: (T,) — 1 в начале сегмента. Возвращает префиксы (M, C)."""
    M, C, F = m.copy(), c.copy(), f.astype(float).copy(); T = len(m); d = 1
    while d < T:
        Mn, Cn, Fn = M.copy(), C.copy(), F.copy()
        keep = (F[d:] > 0)[:, None]
        Mn[d:] = np.where(keep, M[d:], M[d:] * M[:-d])
        Cn[d:] = np.where(keep, C[d:], M[d:] * C[:-d] + C[d:])
        Fn[d:] = np.maximum(F[d:], F[:-d])
        M, C, F = Mn, Cn, Fn; d *= 2
    return M, C

def forward_seq(m, c, starts, k0):
    """Последовательный эталон. starts — индексы начал сегментов; k0: (S, K)."""
    T, K = m.shape; k = np.empty((T, K)); ends = list(starts[1:]) + [T]
    for s, (a, b) in enumerate(zip(starts, ends)):
        x = k0[s]
        for t in range(a, b):
            x = m[t] * x + c[t]; k[t] = x
    return k

def adjoint_seq(m, g, starts):
    """lambda_t = g_t + m_{t+1} lambda_{t+1} внутри сегмента."""
    T, K = m.shape; lam = np.empty((T, K)); ends = list(starts[1:]) + [T]
    for a, b in zip(starts, ends):
        lam[b - 1] = g[b - 1]
        for t in range(b - 2, a - 1, -1): lam[t] = g[t] + m[t + 1] * lam[t + 1]
    return lam
