"""
Сегментированный параллельный scan для аффинной рекурсии k_t = m_t * k_{t-1} + c_t
(истории разной длины в одном массиве) и экономный по памяти обратный проход.

Формат данных: m, c — (T, K); seg_id — (T,) номер истории для каждой позиции (неубывающий);
k0 — (S, K) начальные состояния историй. Пустые истории не допускаются.
"""
import torch

def step_coefficients(u, lam, h):
    """Коэффициенты точного шага по формуле (2): u, lam: (T, K) или broadcast; h: (T, 1)."""
    s = u + lam
    m = torch.exp(-s * h)
    c = (u / s) * (1.0 - m)
    return m, c

def _flags_from_seg(seg_id):
    f = torch.ones_like(seg_id, dtype=torch.bool)
    f[1:] = seg_id[1:] != seg_id[:-1]
    return f

def seg_scan(m, c, flags):
    """Алгоритм удвоения шага (Hillis–Steele): O(T K log T) работы, O(log T) глубины.
    flags: (T,) bool, True в начале сегмента. Возвращает префиксы (M, C) формы (T, K)."""
    M, C, F = m, c, flags
    T = m.shape[0]; d = 1
    while d < T:
        keep = F[d:].unsqueeze(-1)
        Mn = torch.cat([M[:d], torch.where(keep, M[d:], M[d:] * M[:-d])], 0)
        Cn = torch.cat([C[:d], torch.where(keep, C[d:], M[d:] * C[:-d] + C[d:])], 0)
        Fn = torch.cat([F[:d], F[d:] | F[:-d]], 0)
        M, C, F = Mn, Cn, Fn; d *= 2
    return M, C

def chunked_seg_scan(m, c, flags, B=32):
    """Гибридный блочный сегментированный scan: цикл внутри блоков длины B (векторно по блокам),
    сегментированный scan по сводкам блоков, затем коррекция. Работа O(T K), глубина O(B + log(T/B))."""
    T, K = m.shape; nb = -(-T // B); pad = nb * B - T
    one = torch.ones(pad, K, dtype=m.dtype, device=m.device); zero = torch.zeros(pad, K, dtype=c.dtype, device=c.device)
    mp = torch.cat([m, one]).reshape(nb, B, K); cp = torch.cat([c, zero]).reshape(nb, B, K)
    fp = torch.cat([flags, torch.zeros(pad, dtype=torch.bool, device=flags.device)]).reshape(nb, B)
    Ml, Cl, Fl = [], [], []
    M = torch.ones(nb, K, dtype=m.dtype, device=m.device); C = torch.zeros(nb, K, dtype=c.dtype, device=c.device)
    F = torch.zeros(nb, dtype=torch.bool, device=m.device)
    for j in range(B):
        s = fp[:, j].unsqueeze(-1)
        M = torch.where(s, mp[:, j], mp[:, j] * M)
        C = torch.where(s, cp[:, j], mp[:, j] * C + cp[:, j])
        F = F | fp[:, j]
        Ml.append(M); Cl.append(C); Fl.append(F)
    Ml = torch.stack(Ml, 1); Cl = torch.stack(Cl, 1); Fl = torch.stack(Fl, 1)
    PM, PC = seg_scan(Ml[:, -1], Cl[:, -1], Fl[:, -1])
    carryM = torch.cat([torch.ones(1, K, dtype=m.dtype, device=m.device), PM[:-1]])
    carryC = torch.cat([torch.zeros(1, K, dtype=c.dtype, device=c.device), PC[:-1]])
    valid = torch.ones(nb, dtype=torch.bool, device=m.device); valid[0] = False
    use = ((~Fl) & valid.unsqueeze(-1)).unsqueeze(-1)
    Mg = torch.where(use, Ml * carryM.unsqueeze(1), Ml)
    Cg = torch.where(use, Ml * carryC.unsqueeze(1) + Cl, Cl)
    return Mg.reshape(-1, K)[:T], Cg.reshape(-1, K)[:T]

def _scan(m, c, flags, method, B):
    return chunked_seg_scan(m, c, flags, B) if method == "chunked" else seg_scan(m, c, flags)

class SegAffineScan(torch.autograd.Function):
    """Прямой проход — сегментированный scan; обратный — сопряжённая рекурсия (Предложение 2),
    вычисляемая тем же scan на обращённом массиве. Память: O(T K), без хранения уровней scan."""
    @staticmethod
    def forward(ctx, m, c, k0, seg_id, method="hillis", B=32):
        flags = _flags_from_seg(seg_id)
        M, C = _scan(m, c, flags, method, B)
        ctx.method, ctx.B = method, B
        k = M * k0[seg_id] + C
        ctx.save_for_backward(m, k, k0, seg_id)
        return k

    @staticmethod
    def backward(ctx, g):
        m, k, k0, seg_id = ctx.saved_tensors
        T = m.shape[0]
        # конец сегмента: последний элемент или следующий элемент начинает новый сегмент
        is_end = torch.ones(T, dtype=torch.bool, device=m.device)
        is_end[:-1] = seg_id[1:] != seg_id[:-1]
        # коэффициент при lambda_{t+1}: m_{t+1}, если t+1 в том же сегменте, иначе 0
        a = torch.zeros_like(m); a[:-1] = m[1:]
        a = torch.where(is_end.unsqueeze(-1), torch.zeros_like(a), a)
        # обращённый массив: lambda = префикс аффинной рекурсии с начальным значением 0
        Mr, Cr = _scan(a.flip(0), g.flip(0), is_end.flip(0), ctx.method, ctx.B)
        lam = Cr.flip(0)
        # предыдущее состояние: k_{t-1} внутри сегмента, k0 в начале сегмента
        flags = _flags_from_seg(seg_id)
        kprev = torch.empty_like(k); kprev[1:] = k[:-1]
        kprev = torch.where(flags.unsqueeze(-1), k0[seg_id], kprev)
        grad_m = lam * kprev
        grad_c = lam
        grad_k0 = torch.zeros_like(k0).index_add_(0, seg_id[flags], (m * lam)[flags])
        return grad_m, grad_c, grad_k0, None, None, None

def trajectories(m, c, k0, seg_id, memory_efficient=True, method="hillis", B=32):
    """method: "hillis" — удвоение шага; "chunked" — гибридный блочный алгоритм с блоком B."""
    if memory_efficient:
        return SegAffineScan.apply(m, c, k0, seg_id, method, B)
    flags = _flags_from_seg(seg_id)
    M, C = _scan(m, c, flags, method, B)  # autograd через все уровни scan
    return M * k0[seg_id] + C

def sequential(m, c, k0, seg_id):
    """Последовательный точный счёт (базовая линия): цикл по позициям внутри пакета
    историй одинаковой длины не используется — цикл по шагу t по всем историям сразу
    требует выравнивания; здесь — эталонная реализация по позициям."""
    T = m.shape[0]; out = torch.empty_like(m); flags = _flags_from_seg(seg_id)
    x = None
    for t in range(T):
        x = k0[seg_id[t]] if flags[t] else x
        x = m[t] * x + c[t]; out[t] = x
    return out
