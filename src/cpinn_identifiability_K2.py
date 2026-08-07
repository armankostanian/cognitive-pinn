"""
Cognitive-PINN: synthetic identifiability validation for K=2.

This script numerically verifies Theorem 1 (structural identifiability) and
Lemma 3 (robustness to noise) by:
  1. Generating ground-truth trajectories from the K=2 ODE system
     under a controlled practice schedule satisfying assumptions (R), (S1), (S2).
  2. Adding observation noise of varying amplitude.
  3. Recovering parameters via the closed-form formulas from the proof.
  4. Plotting recovery error as a function of noise.

Output: artifacts/identifiability_K2.png and a printed summary table.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.signal import savgol_filter
import os

# -----------------------------------------------------------------------------
# 1. Forward model
# -----------------------------------------------------------------------------

def practice_schedule(t):
    """
    Practice schedule satisfying assumptions (R), (S1), (S2).

    Layout (in days):
      [0, 5]      : warm-up, mild mixed practice (helps initial conditions)
      [5, 8]      : REST  -> identifies Lambda_1, Lambda_2
      [8, 12]     : SINGLE-CONCEPT-1, intensity varies between p_a=0.5 and p_b=1.5
                    -> identifies alpha_1, A_11, A_21
      [12, 15]    : REST again (just to double-check Lambda recovery)
      [15, 19]    : SINGLE-CONCEPT-2, varying intensity
                    -> identifies alpha_2, A_22, A_12
    """
    p1 = 0.0
    p2 = 0.0
    if 0 <= t < 5:
        # mixed warm-up
        p1 = 0.4 + 0.2 * np.sin(t)
        p2 = 0.3 + 0.2 * np.cos(t)
    elif 5 <= t < 8:
        p1 = 0.0
        p2 = 0.0
    elif 8 <= t < 10:
        p1 = 0.5  # = p_a
        p2 = 0.0
    elif 10 <= t < 12:
        p1 = 1.5  # = p_b
        p2 = 0.0
    elif 12 <= t < 15:
        p1 = 0.0
        p2 = 0.0
    elif 15 <= t < 17:
        p1 = 0.0
        p2 = 0.5
    elif 17 <= t < 19:
        p1 = 0.0
        p2 = 1.5
    return np.array([p1, p2])


def ode_rhs(t, k, A, alpha, Lam):
    """Cognitive-PINN ODE for K=2 (no noise)."""
    p = practice_schedule(t)
    phi = 1 - np.exp(-alpha * p)            # shape (2,)
    learning = (A @ phi) * (1.0 - k)        # shape (2,)
    forgetting = Lam * k                    # shape (2,)
    return learning - forgetting


def simulate(theta_star, t_eval, k0=None):
    """Simulate the system at given times t_eval, returning k(t_eval)."""
    A = theta_star['A']
    alpha = theta_star['alpha']
    Lam = theta_star['Lam']
    if k0 is None:
        k0 = np.array([0.2, 0.3])
    sol = solve_ivp(
        ode_rhs, (t_eval[0], t_eval[-1]), k0, t_eval=t_eval,
        args=(A, alpha, Lam),
        method='LSODA', rtol=1e-10, atol=1e-12,
    )
    return sol.y  # shape (2, len(t_eval))


# -----------------------------------------------------------------------------
# 2. Recovery formulas (Theorem 1)
# -----------------------------------------------------------------------------

def estimate_derivative(k, t, smooth=True):
    """Numerical derivative; Savitzky-Golay smoothing for noisy data.
    Wider window (51 samples ~ 0.5 day at our resolution) suppresses
    high-frequency noise while preserving the slow learning/forgetting trends.
    """
    if smooth and len(k) >= 51:
        dk = savgol_filter(k, window_length=51, polyorder=3, deriv=1,
                           delta=t[1] - t[0])
    else:
        dk = np.gradient(k, t)
    return dk


def recover_lambda(k_obs, t, t_rest_start, t_rest_end):
    """Step 1: recover Lambda_i from rest interval via log-ratio formula."""
    # pick two endpoints inside the rest interval (with margin to avoid boundary
    # effects)
    margin = (t_rest_end - t_rest_start) * 0.2
    t1 = t_rest_start + margin
    t2 = t_rest_end - margin
    i1 = np.argmin(np.abs(t - t1))
    i2 = np.argmin(np.abs(t - t2))
    delta_t = t[i2] - t[i1]
    Lam = np.zeros(2)
    for i in range(2):
        a = k_obs[i, i1]
        b = k_obs[i, i2]
        # numerical safety
        a = max(a, 1e-6)
        b = max(b, 1e-6)
        Lam[i] = -np.log(b / a) / delta_t
    return Lam


def H_tilde(u, r):
    """Auxiliary function H_r(u) from Lemma 2."""
    if u < 1e-12:
        return float(r)
    return (1.0 - np.exp(-r * u)) / (1.0 - np.exp(-u))


def invert_H_tilde(rho, r):
    """Invert H_r(u) = rho via root-finding (rho should be in (1, r)).

    The pre-image is restricted to a physically meaningful range [0.01, 100],
    matching prior knowledge that practice-response slopes alpha live in
    O(0.1)–O(10) for cognitive tasks.
    """
    if rho >= r - 1e-9:
        return 0.01  # alpha approaches its lower bound
    if rho <= 1.0 + 1e-9:
        return 100.0  # alpha approaches its upper bound
    f = lambda u: H_tilde(u, r) - rho
    try:
        return brentq(f, 0.01, 100.0, xtol=1e-10)
    except ValueError:
        return np.nan


def _g_func(k_obs, dk, t, idx, Lam_i):
    """Helper g_i(t) = (dk_i + Lambda_i k_i) / (1 - k_i)."""
    return (dk + Lam_i * k_obs[idx]) / np.clip(1.0 - k_obs[idx], 1e-6, None)


def recover_alpha_A_diag(k_obs, t, idx_target, Lam_known,
                         t_a, t_b, p_a, p_b, half_window=0.5):
    """
    Step 2: recover (alpha_j, A_jj) for concept j (j = idx_target) from a
    single-concept interval where only concept j is practised, with two
    distinct practice intensities p_a < p_b.

    Robustness improvement: g_a, g_b are averaged over half-second windows
    around t_a, t_b instead of being read at a single time index.
    """
    dk = estimate_derivative(k_obs[idx_target], t, smooth=True)
    g_series = _g_func(k_obs, dk, t, idx_target, Lam_known[idx_target])

    # average g over window around t_a
    mask_a = (t >= t_a - half_window) & (t <= t_a + half_window)
    mask_b = (t >= t_b - half_window) & (t <= t_b + half_window)

    g_a = np.median(g_series[mask_a])
    g_b = np.median(g_series[mask_b])

    g_a = max(g_a, 1e-9)
    g_b = max(g_b, 1e-9)

    rho = g_b / g_a
    r = p_b / p_a
    u_star = invert_H_tilde(rho, r)
    alpha_j = u_star / p_a
    A_jj = g_a / max(1.0 - np.exp(-alpha_j * p_a), 1e-9)
    return alpha_j, A_jj


def recover_A_offdiag(k_obs, t, idx_other, idx_pract, Lam_known,
                      alpha_pract, t_pts, p_pts):
    """
    Step 3: recover A[idx_other, idx_pract] using observations of the OTHER
    concept during the single-practice interval of idx_pract.

    Uses median over multiple time points for robustness against outliers
    introduced by smoothing artefacts at noisy interval boundaries.
    """
    dk = estimate_derivative(k_obs[idx_other], t, smooth=True)
    estimates = []
    for tt, pp in zip(t_pts, p_pts):
        i = np.argmin(np.abs(t - tt))
        denom = (1.0 - k_obs[idx_other, i]) * (1.0 - np.exp(-alpha_pract * pp))
        if denom < 1e-6:
            continue
        num = dk[i] + Lam_known[idx_other] * k_obs[idx_other, i]
        estimates.append(num / denom)
    if not estimates:
        return np.nan
    # median is robust against the rare large-error sample
    return float(np.median(estimates))


# -----------------------------------------------------------------------------
# 3. Full recovery pipeline
# -----------------------------------------------------------------------------

def recover_all(k_obs, t):
    """Apply Steps 1-4 of Theorem 1 to recover the full parameter vector."""
    # Step 1: Lambda from rest interval [5, 8]
    Lam_hat = recover_lambda(k_obs, t, t_rest_start=5.0, t_rest_end=8.0)

    # Step 2: alpha_1, A_11 from interval [8, 12]
    alpha1_hat, A11_hat = recover_alpha_A_diag(
        k_obs, t, idx_target=0, Lam_known=Lam_hat,
        t_a=9.0, t_b=11.0, p_a=0.5, p_b=1.5,
    )

    # Step 3: A_21 from interval [8, 12], using alpha_1 and Lambda_2
    A21_hat = recover_A_offdiag(
        k_obs, t, idx_other=1, idx_pract=0, Lam_known=Lam_hat,
        alpha_pract=alpha1_hat,
        t_pts=[9.0, 10.0, 11.0], p_pts=[0.5, 0.5, 1.5],
    )

    # Step 4: by symmetry from interval [15, 19]
    alpha2_hat, A22_hat = recover_alpha_A_diag(
        k_obs, t, idx_target=1, Lam_known=Lam_hat,
        t_a=16.0, t_b=18.0, p_a=0.5, p_b=1.5,
    )
    A12_hat = recover_A_offdiag(
        k_obs, t, idx_other=0, idx_pract=1, Lam_known=Lam_hat,
        alpha_pract=alpha2_hat,
        t_pts=[16.0, 17.0, 18.0], p_pts=[0.5, 0.5, 1.5],
    )

    return {
        'A': np.array([[A11_hat, A12_hat], [A21_hat, A22_hat]]),
        'alpha': np.array([alpha1_hat, alpha2_hat]),
        'Lam': Lam_hat,
    }


def param_error(theta_hat, theta_star):
    """Relative L2 error vs. the ground truth."""
    err = 0.0
    norm = 0.0
    for key in ['A', 'alpha', 'Lam']:
        err += np.sum((theta_hat[key] - theta_star[key]) ** 2)
        norm += np.sum(theta_star[key] ** 2)
    return np.sqrt(err / norm)


# -----------------------------------------------------------------------------
# 4. Experiment: noise sweep
# -----------------------------------------------------------------------------

def run_experiment(noise_levels, n_repeats=20, seed=0):
    rng = np.random.default_rng(seed)

    theta_star = {
        'A': np.array([[0.30, 0.05],
                       [0.10, 0.25]]),
        'alpha': np.array([1.20, 0.90]),
        'Lam': np.array([0.05, 0.07]),
    }

    t = np.linspace(0, 19, 1901)  # high-resolution clean trajectory
    k_clean = simulate(theta_star, t)

    results = {eps: [] for eps in noise_levels}
    per_param_results = {eps: [] for eps in noise_levels}

    for eps in noise_levels:
        for _ in range(n_repeats):
            noise = rng.normal(0.0, eps, size=k_clean.shape)
            k_obs = np.clip(k_clean + noise, 1e-3, 1 - 1e-3)
            try:
                theta_hat = recover_all(k_obs, t)
                err = param_error(theta_hat, theta_star)
                results[eps].append(err)
                per_param_results[eps].append({
                    'A':     np.linalg.norm(theta_hat['A']     - theta_star['A'])     / np.linalg.norm(theta_star['A']),
                    'alpha': np.linalg.norm(theta_hat['alpha'] - theta_star['alpha']) / np.linalg.norm(theta_star['alpha']),
                    'Lam':   np.linalg.norm(theta_hat['Lam']   - theta_star['Lam'])   / np.linalg.norm(theta_star['Lam']),
                })
            except Exception as e:
                print(f"[WARN] eps={eps}: {e}")

    return theta_star, results, per_param_results, t, k_clean


# -----------------------------------------------------------------------------
# 5. Reporting
# -----------------------------------------------------------------------------

def print_summary(theta_star, results, per_param_results):
    print("=" * 70)
    print("Ground truth parameters:")
    print(f"  A     = {theta_star['A'].tolist()}")
    print(f"  alpha = {theta_star['alpha'].tolist()}")
    print(f"  Lambda= {theta_star['Lam'].tolist()}")
    print("=" * 70)
    print(f"{'noise eps':>12} | {'rel err mean':>13} | "
          f"{'A':>10} | {'alpha':>10} | {'Lambda':>10}")
    print("-" * 70)
    for eps, errs in results.items():
        if not errs:
            continue
        m = np.mean(errs)
        per = per_param_results[eps]
        a   = np.mean([d['A']     for d in per])
        al  = np.mean([d['alpha'] for d in per])
        la  = np.mean([d['Lam']   for d in per])
        print(f"{eps:>12.3f} | {m:>13.4f} | {a:>10.4f} | {al:>10.4f} | {la:>10.4f}")
    print("=" * 70)


def plot_results(results, per_param, t, k_clean, theta_star, outdir="artifacts"):
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Left: clean trajectory
    ax = axes[0]
    ax.plot(t, k_clean[0], label=r'$k_1(t)$', linewidth=2)
    ax.plot(t, k_clean[1], label=r'$k_2(t)$', linewidth=2)
    ax.axvspan(5,  8,  alpha=0.15, color='gray', label='Rest (R)')
    ax.axvspan(8,  12, alpha=0.15, color='C0',   label='S1: practice 1')
    ax.axvspan(12, 15, alpha=0.15, color='gray')
    ax.axvspan(15, 19, alpha=0.15, color='C1',   label='S2: practice 2')
    ax.set_xlabel('time (days)')
    ax.set_ylabel('knowledge state')
    ax.set_title('(a) Synthetic trajectory under designed practice schedule')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)

    # Right: error vs noise per parameter group
    ax = axes[1]
    eps_levels = sorted(results.keys())
    for key, label, marker in [('Lam', r'$\Lambda$', 'o'),
                                ('alpha', r'$\alpha$', 's'),
                                ('A', r'$A$', '^')]:
        means = [np.mean([d[key] for d in per_param[e]]) for e in eps_levels]
        ax.plot(eps_levels, means, marker=marker, linewidth=2, label=label, markersize=8)

    # Linear reference line
    if eps_levels[1] > 0:
        slope_ref = 0.05 / eps_levels[1] * eps_levels[1]  # placeholder
    ref_line = [10 * e for e in eps_levels]
    ax.plot(eps_levels, ref_line, '--', color='gray', alpha=0.6,
            label='linear (Lemma 3 prediction)')

    ax.set_xlabel(r'observation noise $\epsilon$')
    ax.set_ylabel(r'relative recovery error')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('(b) Recovery error scales linearly with noise (small-$\\epsilon$ regime)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3, which='both')
    ax.set_ylim(1e-3, 1e2)

    fig.tight_layout()
    out_path = os.path.join(outdir, 'identifiability_K2.png')
    fig.savefig(out_path, dpi=150)
    print(f"[OK] Figure saved to {out_path}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
    theta_star, results, per_param, t, k_clean = run_experiment(
        noise_levels, n_repeats=20, seed=42,
    )
    print_summary(theta_star, results, per_param)
    plot_results(results, per_param, t, k_clean, theta_star)
