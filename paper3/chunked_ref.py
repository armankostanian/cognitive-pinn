"""Гибридный блочный сегментированный scan (эталон NumPy):
1) массив режется на блоки длины B; внутри блока — последовательный цикл (векторно по всем блокам),
   дающий локальные префиксы (M_loc, C_loc, F_loc) с обнулением на началах сегментов;
2) сводки блоков (последний локальный префикс) — сегментированный scan по T/B элементам;
3) коррекция: позиции блока до первого начала сегмента получают перенос из предыдущих блоков.
Работа O(T K), последовательная глубина O(B + log(T/B))."""
import numpy as np
from ref_numpy import seg_scan

def chunked_scan(m, c, f, B):
    T, K = m.shape; nb = -(-T // B); pad = nb * B - T
    mp = np.concatenate([m, np.ones((pad, K))]).reshape(nb, B, K)
    cp = np.concatenate([c, np.zeros((pad, K))]).reshape(nb, B, K)
    fp = np.concatenate([f, np.zeros(pad)]).reshape(nb, B)
    Ml = np.empty_like(mp); Cl = np.empty_like(cp); Fl = np.empty_like(fp)
    M = np.ones((nb, K)); C = np.zeros((nb, K)); F = np.zeros(nb)
    for j in range(B):                                   # цикл внутри блока, векторно по блокам
        s = fp[:, j] > 0
        M = np.where(s[:, None], mp[:, j], mp[:, j] * M)
        C = np.where(s[:, None], cp[:, j], mp[:, j] * C + cp[:, j])
        F = np.maximum(F, fp[:, j])
        Ml[:, j], Cl[:, j], Fl[:, j] = M, C, F
    Mb, Cb, Fb = Ml[:, -1], Cl[:, -1], Fl[:, -1]           # сводки блоков
    # эксклюзивный префикс по блокам: перенос в блок b = префикс блоков 0..b-1
    PM, PC = seg_scan(Mb, Cb, Fb)                          # инклюзивный
    carryM = np.vstack([np.ones((1, K)), PM[:-1]]); carryC = np.vstack([np.zeros((1, K)), PC[:-1]])
    carry_valid = np.r_[False, np.ones(nb - 1, bool)]
    use = (~(Fl > 0)) & carry_valid[:, None]               # до первого начала сегмента в блоке
    Mg = np.where(use[..., None], Ml * carryM[:, None], Ml)
    Cg = np.where(use[..., None], Ml * carryC[:, None] + Cl, Cl)
    return Mg.reshape(-1, K)[:T], Cg.reshape(-1, K)[:T]
