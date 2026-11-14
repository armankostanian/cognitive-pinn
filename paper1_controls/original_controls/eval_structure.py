"""
Core result of the repositioned paper: does the recovered Cognitive-PINN
transfer matrix A reproduce the EXPERT KC prerequisite graph?

Compares the off-diagonal magnitudes of A against the binary expert graph G:
  - edge ROC-AUC and average precision (does |A_ij| rank true prereq edges high?)
  - precision@k
  - a NULL baseline: same metrics for random graphs of identical density, to
    show the match is non-trivial (mean +/- sd over --null_reps draws), plus an
    empirical p-value.
Optionally correlates |A| with crowd-sourced edge strengths (0-9) if provided.

You provide A as a .npy (K x K), saved from your trained CPINN on Junyi:
    np.save("A_junyi.npy", model.A().detach().cpu().numpy())

Run:
  python eval_structure.py --A A_junyi.npy --graph clean/junyi/kc_prereq_graph.npy
"""
from __future__ import annotations
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def offdiag(M):
    K = M.shape[0]
    mask = ~np.eye(K, dtype=bool)
    return M[mask]


def precision_at_k(scores, labels, k):
    order = np.argsort(-scores)
    return labels[order][:k].sum() / max(k, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", required=True, help="recovered transfer matrix (KxK .npy)")
    ap.add_argument("--graph", required=True, help="expert KC prereq graph (KxK .npy)")
    ap.add_argument("--crowd", default=None, help="optional KxK crowd strength .npy")
    ap.add_argument("--null_reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    A = np.load(a.A); G = np.load(a.graph)
    assert A.shape == G.shape, f"shape mismatch {A.shape} vs {G.shape}"
    K = A.shape[0]
    s = np.abs(offdiag(A)).astype(float)        # edge scores from |A|
    y = (offdiag(G) > 0).astype(int)            # true expert edges
    dens = y.mean()
    n_edges = int(y.sum())
    print(f"K={K}  off-diag pairs={len(y)}  expert edges={n_edges}  density={dens:.3f}")
    if n_edges == 0 or n_edges == len(y):
        print("!! degenerate reference graph; adjust thresholds in junyi_prereq_graph.py")
        return

    auc = roc_auc_score(y, s)
    ap_ = average_precision_score(y, s)
    p_at = {k: precision_at_k(s, y, k) for k in (n_edges, 2 * n_edges, 20)}
    print(f"\nRecovered A vs expert graph:")
    print(f"  edge ROC-AUC      = {auc:.4f}")
    print(f"  average precision = {ap_:.4f}  (chance = density = {dens:.3f})")
    for k, v in p_at.items():
        print(f"  precision@{k:<5}   = {v:.4f}")

    # null: random graphs of identical density (AUC is invariant to relabeling
    # of y with fixed count -> permutation test on labels)
    rng = np.random.default_rng(a.seed)
    null = np.empty(a.null_reps)
    for i in range(a.null_reps):
        yp = np.zeros_like(y); yp[rng.choice(len(y), n_edges, replace=False)] = 1
        null[i] = roc_auc_score(yp, s)
    p = (null >= auc).mean()
    print(f"\nNull (random graphs, same density, {a.null_reps} draws):")
    print(f"  AUC null mean={null.mean():.4f} sd={null.std():.4f} "
          f"95%=[{np.percentile(null,2.5):.4f},{np.percentile(null,97.5):.4f}]")
    print(f"  empirical p(AUC_obs <= null) = {p:.4g}  "
          f"({'significant' if p < 0.05 else 'NOT significant'} vs random)")

    if a.crowd:
        C = np.load(a.crowd)
        cs = offdiag(C).astype(float); m = cs > 0
        if m.sum() > 2:
            from scipy.stats import spearmanr
            rho, pp = spearmanr(np.abs(offdiag(A))[m], cs[m])
            print(f"\nCrowd strength (0-9) correlation: Spearman rho={rho:.3f} p={pp:.2e} "
                  f"(n={int(m.sum())} rated pairs)")


if __name__ == "__main__":
    main()
