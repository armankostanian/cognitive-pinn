"""
Замеры для статьи 3 (синтетика, Э1–Э6). Запуск:
    python bench.py --exp all --device cuda --out results.jsonl
    python bench.py --exp e2 --device cpu --out results_cpu.jsonl
Каждый замер: 2 прогрева + 5 повторов, медиана и разброс; память — пик на GPU.
Э7 (Junyi) и Э8 (одно обновление на устройстве) — см. ЗАДАНИЕ_статья3.md.
"""
import argparse, json, math, platform, statistics, time
import numpy as np, torch
import scan_torch as S

def sync(dev):
    if dev == "cuda": torch.cuda.synchronize()

def timeit(fn, dev, reps=5, warm=2):
    for _ in range(warm): fn(); sync(dev)
    ts = []
    for _ in range(reps):
        sync(dev); t0 = time.perf_counter(); fn(); sync(dev); ts.append(time.perf_counter() - t0)
    return statistics.median(ts), (max(ts) - min(ts))

def peak_mem(fn, dev):
    if dev != "cuda": return None
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); fn(); sync(dev)
    return torch.cuda.max_memory_allocated() / 2**20      # МБ

def make_data(N, K, lens, dev, dtype, seed=0):
    """Истории длины lens[n]; u — практика на случайной теме в 50% интервалов; h ∈ {1/24,1,3,14} сут."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    T = int(sum(lens))
    lam = (torch.rand(K, generator=g) * 0.3 + 0.02).to(dtype)
    u = torch.zeros(T, K, dtype=dtype)
    prac = torch.rand(T, generator=g) < 0.5
    topic = torch.randint(0, K, (T,), generator=g)
    u[prac, topic[prac]] = (torch.rand(int(prac.sum()), generator=g) * 4 + 1).to(dtype)
    h = torch.tensor([1/24, 1.0, 3.0, 14.0], dtype=dtype)[torch.randint(0, 4, (T,), generator=g)]
    h = torch.where(prac, torch.full_like(h, 1/24), h).unsqueeze(-1)
    m, c = S.step_coefficients(u, lam, h)
    seg = torch.repeat_interleave(torch.arange(N), torch.as_tensor(lens))
    k0 = (torch.rand(N, K, generator=g) * 0.6 + 0.2).to(dtype)
    return [x.to(dev) for x in (m, c, k0, seg, u, h)] + [lam.to(dev)]

def to_padded(x, lens, fill):
    N, L = len(lens), int(max(lens)); out = torch.full((N, L) + x.shape[1:], fill, dtype=x.dtype, device=x.device)
    idx = torch.repeat_interleave(torch.arange(N, device=x.device), torch.as_tensor(lens, device=x.device))
    pos = torch.cat([torch.arange(l, device=x.device) for l in lens])
    out[idx, pos] = x
    return out

def seq_padded(mP, cP, k0):
    x = k0; out = torch.empty_like(mP)
    for t in range(mP.shape[1]):
        x = mP[:, t] * x + cP[:, t]; out[:, t] = x
    return out

def rk4_padded(uP, hP, lam, k0, hmax=0.25):
    x = k0; out = torch.empty_like(uP)
    for t in range(uP.shape[1]):
        n = max(1, math.ceil(float(hP[:, t].max()) / hmax)); d = hP[:, t] / n; u = uP[:, t]
        f = lambda y: u * (1 - y) - lam * y
        for _ in range(n):
            a1 = f(x); a2 = f(x + d / 2 * a1); a3 = f(x + d / 2 * a2); a4 = f(x + d * a3)
            x = x + d / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        out[:, t] = x
    return out

def record(out, **kw):
    kw.update(device_name=torch.cuda.get_device_name(0) if kw.get("device") == "cuda" else platform.processor(),
              torch=torch.__version__)
    out.write(json.dumps(kw) + "\n"); out.flush(); print(kw)

def e1(dev, out):
    """Точность (float64): scan против последовательного счёта и RK4."""
    N, K = 64, 5; lens = [200] * N
    m, c, k0, seg, u, h, lam = make_data(N, K, lens, dev, torch.float64)
    k_scan = S.trajectories(m, c, k0, seg)
    mP, cP = to_padded(m, lens, 1.0), to_padded(c, lens, 0.0)
    k_seq = seq_padded(mP, cP, k0).reshape(-1, K)
    k_rk4 = rk4_padded(to_padded(u, lens, 0.0), to_padded(h, lens, 0.0), lam, k0).reshape(-1, K)
    record(out, exp="e1", device=dev, N=N, K=K, L=200,
           dev_scan_vs_seq=float((k_scan - k_seq).abs().max()), dev_rk4_vs_seq=float((k_rk4 - k_seq).abs().max()))

def run_methods(N, K, lens, dev, dtype, out, exp, methods):
    m, c, k0, seg, u, h, lam = make_data(N, K, lens, dev, dtype)
    mP, cP = to_padded(m, lens, 1.0), to_padded(c, lens, 0.0)
    flagsP = torch.repeat_interleave(torch.arange(N, device=dev), mP.shape[1])
    fns = {
        "seq_padded": lambda: seq_padded(mP, cP, k0),
        "scan_padded": lambda: S.trajectories(mP.reshape(-1, K), cP.reshape(-1, K), k0, flagsP),
        "scan_segmented": lambda: S.trajectories(m, c, k0, seg),
        "scan_chunked": lambda: S.trajectories(m, c, k0, seg, method="chunked", B=32),
        "rk4_padded": lambda: rk4_padded(to_padded(u, lens, 0.0), to_padded(h, lens, 0.0), lam, k0),
    }
    for name in methods:
        try:
            tmed, tspread = timeit(fns[name], dev)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); record(out, exp=exp, device=dev, method=name, N=N, K=K, T=int(sum(lens)), skipped="OOM"); continue
        record(out, exp=exp, device=dev, method=name, N=N, K=K, T=int(sum(lens)), L_max=int(max(lens)),
               L_mean=float(np.mean(lens)), time_s=tmed, spread_s=tspread, peak_MB=peak_mem(fns[name], dev))

def e2(dev, out):
    """Масштабирование по числу учеников (K=10, L=100)."""
    for N in (10**3, 10**4, 10**5):
        run_methods(N, 10, [100] * N, dev, torch.float32, out, "e2",
                    ["seq_padded", "scan_segmented", "scan_chunked"] + (["rk4_padded"] if N <= 10**4 else []))

def e3(dev, out):
    """Масштабирование по длине истории и числу тем."""
    for L in (100, 1000, 10000):
        N = max(10, 10**6 // L)
        run_methods(N, 10, [L] * N, dev, torch.float32, out, "e3_L", ["seq_padded", "scan_segmented", "scan_chunked"])
    for K in (2, 10, 50, 100):
        run_methods(1000, K, [500] * 1000, dev, torch.float32, out, "e3_K", ["seq_padded", "scan_segmented", "scan_chunked"])

def e4(dev, out):
    """Разная длина историй (логнормальное распределение, как в журналах): сегментирование против дополнения."""
    rng = np.random.default_rng(4)
    for N in (10**3, 10**4):
        lens = np.clip(rng.lognormal(mean=3.5, sigma=1.2, size=N).astype(int), 1, 5000).tolist()
        run_methods(N, 10, lens, dev, torch.float32, out, "e4", ["seq_padded", "scan_padded", "scan_segmented", "scan_chunked"])

def e5(dev, out):
    """Градиенты: обратный scan (экономный по памяти) против autograd через уровни scan."""
    for N, L in ((1000, 1000), (10000, 1000)):
        m, c, k0, seg, *_ = make_data(N, 10, [L] * N, dev, torch.float32)
        w = torch.rand_like(m)
        for eff, meth in ((True, "hillis"), (True, "chunked"), (False, "hillis")):
            def f():
                a, b = m.clone().requires_grad_(), c.clone().requires_grad_()
                (w * S.trajectories(a, b, k0, seg, memory_efficient=eff, method=meth) ** 2).sum().backward()
            try:
                tmed, tspread = timeit(f, dev, reps=3)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); record(out, exp="e5", device=dev, method=meth, N=N, L=L, skipped="OOM"); continue
            record(out, exp="e5", device=dev, method=("adjoint_" if eff else "autograd_") + meth, N=N, L=L, K=10,
                   time_s=tmed, spread_s=tspread, peak_MB=peak_mem(f, dev))

def e6(dev, out):
    """Маргинальное правдоподобие: N учеников × Q узлов квадратуры (одна оценка целевой функции)."""
    for N in (100, 1000, 10000):
        for Q in (9, 81, 625):
            if N * Q > 2_000_000: continue
            m, c, k0, seg, *_ = make_data(N * Q, 2, [20] * (N * Q), dev, torch.float32)
            y = (torch.rand_like(m) < 0.6).float()
            def f():
                k = S.trajectories(m, c, k0, seg)
                w = 5.0 * k - 2.0
                return (y * torch.nn.functional.logsigmoid(w) + (1 - y) * torch.nn.functional.logsigmoid(-w)).sum()
            tmed, tspread = timeit(f, dev)
            record(out, exp="e6", device=dev, N=N, Q=Q, T_per_learner=20, time_s=tmed, spread_s=tspread,
                   peak_MB=peak_mem(f, dev))

def e9(dev, out):
    """Подбор размера блока гибридного алгоритма: равные и разные длины историй."""
    rng = np.random.default_rng(9)
    cases = [("uniform_L100", [100] * 10**4), ("uniform_L1000", [1000] * 1000),
             ("lognormal", np.clip(rng.lognormal(3.5, 1.2, 10**4).astype(int), 1, 5000).tolist())]
    for name, lens in cases:
        m, c, k0, seg, *_ = make_data(len(lens), 10, lens, dev, torch.float32)
        for B in (8, 16, 32, 64, 128, 256):
            tmed, tspread = timeit(lambda: S.trajectories(m, c, k0, seg, method="chunked", B=B), dev)
            record(out, exp="e9", device=dev, case=name, B=B, T=int(sum(lens)), time_s=tmed, spread_s=tspread)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results.jsonl")
    a = ap.parse_args()
    exps = {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5, "e6": e6, "e9": e9}
    with open(a.out, "a") as out:
        for name in (exps if a.exp == "all" else a.exp.split(",")):
            exps[name](a.device, out)
