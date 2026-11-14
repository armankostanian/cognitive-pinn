"""Численная проверка теоретического ядра статьи №2.
Модель: dk_i/dt = u_i(t)(1-k_i) - Lam_i k_i,  u = A phi(p),  phi_j(p) = 1 - exp(-alpha_j p)."""
import numpy as np
from scipy.optimize import brentq
rng = np.random.default_rng(42)

def step(k, u, Lam, h):                       # Лемма 1: точный шаг
    s = u + Lam; ks = u / s
    return ks + (k - ks) * np.exp(-s * h)

def u_of(A, alpha, p):
    return A @ (1 - np.exp(-alpha * p))

# ---------- Теорема 2: восстановление по дискретным замерам, общее K ----------
def simulate_design(A, Lam, alpha, k0, pa=4.0, pb=10.0, hs=1/24, hrest=3.0):
    """Для каждой темы j: изолированные занятия с интенсивностью pa и pb (замер до/после),
    между ними паузы (замеры в начале и конце). Возвращает список наблюдений."""
    K = len(Lam); k = k0.copy(); obs = []
    for j in range(K):
        for p in (pa, pb):
            # пауза
            k_start = k.copy(); k = step(k, np.zeros(K), Lam, hrest)
            obs.append(("rest", None, None, hrest, k_start, k.copy()))
            # изолированное занятие темы j
            pv = np.zeros(K); pv[j] = p
            k_start = k.copy(); k = step(k, u_of(A, alpha, pv), Lam, hs)
            obs.append(("sess", j, p, hs, k_start, k.copy()))
    return obs

def reconstruct(obs, K, pa=4.0, pb=10.0):
    # шаг 1: Lambda по паузам
    Lam = np.mean([-np.log(o[5] / o[4]) / o[3] for o in obs if o[0] == "rest"], axis=0)
    # шаг 2: u_i по каждой сессии — обращение монотонной функции k(h; u)
    U = {}
    for kind, j, p, h, ks, ke in obs:
        if kind != "sess": continue
        u = np.zeros(K)
        for i in range(K):
            f = lambda x: step(np.array([ks[i]]), np.array([x]), np.array([Lam[i]]), h)[0] - ke[i]
            u[i] = 0.0 if abs(f(0.0)) < 1e-13 else brentq(f, 0.0, 1e4, xtol=1e-14)
        U[(j, p)] = u
    # шаг 3: alpha_j по отношению откликов (лемма о монотонной сигнатуре), затем столбец A
    r = pb / pa; A = np.zeros((K, K)); alpha = np.zeros(K)
    for j in range(K):
        ua, ub = U[(j, pa)], U[(j, pb)]
        i = int(np.argmax(ua)); rho = ub[i] / ua[i]
        H = lambda x: (1 - np.exp(-r * x)) / (1 - np.exp(-x)) - rho
        alpha[j] = brentq(H, 1e-10, 200.0, xtol=1e-15) / pa
        A[:, j] = ua / (1 - np.exp(-alpha[j] * pa))
    return A, Lam, alpha

worst = 0.0
for K in (2, 3, 5, 10):
    for trial in range(20):
        A = np.abs(rng.normal(3, 1.5, (K, K))) * (rng.random((K, K)) < 0.7); np.fill_diagonal(A, np.abs(rng.normal(4, 1, K)))
        Lam = rng.uniform(0.01, 0.5, K); alpha = rng.uniform(0.05, 0.4, K); k0 = rng.uniform(0.2, 0.8, K)
        obs = simulate_design(A, Lam, alpha, k0)
        A_, L_, a_ = reconstruct(obs, K)
        err = max(np.abs(A_ - A).max() / A.max(), np.abs(L_ - Lam).max() / Lam.max(), np.abs(a_ - alpha).max() / alpha.max())
        worst = max(worst, err)
print(f"Теорема 2 (общее K = 2,3,5,10; по 20 случайных наборов): макс. отн. ошибка восстановления = {worst:.1e}")

# ---------- Теорема 3: необходимость вариации интенсивности ----------
K = 3; A = np.array([[4.0, 0.8, 0.0], [1.2, 3.0, 0.5], [0.3, 0.9, 5.0]])
Lam = np.array([0.05, 0.2, 0.1]); alpha = np.array([0.12, 0.25, 0.08]); pbar = np.array([6.0, 6.0, 6.0])
alpha2 = np.array([0.40, 0.03, 0.70])                     # совсем другие alpha
phi = lambda a, p: 1 - np.exp(-a * p)
A2 = A * (phi(alpha, pbar) / phi(alpha2, pbar))[None, :]   # преобразование из теоремы 3
def run(Ax, ax, schedule):
    k = np.full(K, 0.4); out = []
    for pv, h in schedule:
        k = step(k, u_of(Ax, ax, pv), Lam, h); out.append(k.copy())
    return np.array(out)
sched_const, sched_var = [], []
for _ in range(60):
    j = rng.integers(K); pv = np.zeros(K)
    if rng.random() < 0.6:
        if rng.random() < 0.5: pv[j] = pbar[j]
        else: pv[:] = pbar * (rng.random(K) < 0.5)        # и совместная практика тоже
    h = rng.choice([1/24, 1.0, 3.0, 14.0])
    sched_const.append((pv, h))
    pv2 = pv.copy(); pv2[pv2 > 0] = rng.choice([3.0, 6.0, 12.0], size=(pv2 > 0).sum())
    sched_var.append((pv2, h))
d_const = np.abs(run(A, alpha, sched_const) - run(A2, alpha2, sched_const)).max()
d_var = np.abs(run(A, alpha, sched_var) - run(A2, alpha2, sched_var)).max()
print(f"Теорема 3: постоянная интенсивность — расхождение траекторий двух разных (A,alpha): {d_const:.1e}")
print(f"           переменная интенсивность — расхождение тех же наборов: {d_var:.2f}")
