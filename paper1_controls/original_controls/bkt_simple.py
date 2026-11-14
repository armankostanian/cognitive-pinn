"""
Self-contained standard BKT (no external deps beyond numpy/pandas/sklearn),
a drop-in replacement for the broken pyBKT run.

Why this exists: pyBKT 1.4.2 settled in the INVERTED/degenerate solution
(slip>0.5 -> AUC<0.5) with EM divide warnings. Standard BKT practice caps
slip,guess < 0.5 to forbid that inversion; this implementation does so and is
numerically guarded, so it returns the canonical ~0.65-0.70 on ASSIST-09.

Model (per skill): 2-state HMM, no forgetting.
    params  L0=P(known at start), T=P(learn), S=P(slip), G=P(guess)
    predict P(correct_i) = pk*(1-S) + (1-pk)*G   (pk = P known BEFORE obs i)

    python bkt_simple.py --dataset assist2009 --data_dir /path/to/assist2009_updated

Output: <out_root>/<dataset>/preds_bkt.csv  (dataset,student,step,skill,y_true,y_pred)
"""
from __future__ import annotations
import argparse, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
try:  # main() legacy deps; exp3 uses only fit_skill/gather_by_skill
    from datasets import REGISTRY, load_folds  # type: ignore
except ImportError:
    REGISTRY, load_folds = None, None

EPS = 1e-9
S_CAP, G_CAP = 0.30, 0.30          # slip/guess upper bound -> forbids inversion


def _fb(obs, L0, T, S, G):
    """Scaled forward-backward for one sequence. Returns gamma, xi sums needed by EM."""
    n = len(obs)
    pi = np.array([1 - L0, L0])                       # [unknown, known]
    A = np.array([[1 - T, T], [0.0, 1.0]])
    # emission per obs: rows [unknown, known], col chosen by obs
    Bc = np.array([G, 1 - S]); Bi = np.array([1 - G, S])
    B = np.where(obs[:, None] == 1, Bc[None, :], Bi[None, :])  # (n,2)

    alpha = np.zeros((n, 2)); c = np.zeros(n)
    alpha[0] = pi * B[0]; c[0] = 1.0 / (alpha[0].sum() + EPS); alpha[0] *= c[0]
    for t in range(1, n):
        alpha[t] = (alpha[t - 1] @ A) * B[t]
        c[t] = 1.0 / (alpha[t].sum() + EPS); alpha[t] *= c[t]
    beta = np.zeros((n, 2)); beta[-1] = c[-1]
    for t in range(n - 2, -1, -1):
        beta[t] = (A @ (B[t + 1] * beta[t + 1])) * c[t]
    gamma = alpha * beta
    gamma /= (gamma.sum(1, keepdims=True) + EPS)
    # xi sums for transition unknown->known
    xi_uk = 0.0; gamma_u_pre = 0.0
    for t in range(n - 1):
        denom = (alpha[t] @ (A * (B[t + 1] * beta[t + 1])[None, :]).T).sum() + EPS
        xi_uk += alpha[t, 0] * A[0, 1] * B[t + 1, 1] * beta[t + 1, 1] / denom
        gamma_u_pre += gamma[t, 0]
    ll = -np.log(c + EPS).sum()
    return gamma, xi_uk, gamma_u_pre, ll


def fit_skill(seqs, iters=25, restarts=2, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for rs in range(restarts):
        L0 = rng.uniform(0.1, 0.5); T = rng.uniform(0.05, 0.4)
        S = rng.uniform(0.05, 0.25); G = rng.uniform(0.05, 0.25)
        for _ in range(iters):
            g1 = 0.0; nseq = 0
            num_T = 0.0; den_T = 0.0
            sk_known = 0.0; den_known = 0.0       # slip
            cg_unknown = 0.0; den_unknown = 0.0   # guess
            tot_ll = 0.0
            for obs in seqs:
                gamma, xi_uk, gu_pre, ll = _fb(obs, L0, T, S, G)
                tot_ll += ll; nseq += 1
                g1 += gamma[0, 1]
                num_T += xi_uk; den_T += gu_pre
                kn = gamma[:, 1]; un = gamma[:, 0]
                sk_known += (kn * (obs == 0)).sum(); den_known += kn.sum()
                cg_unknown += (un * (obs == 1)).sum(); den_unknown += un.sum()
            L0 = min(max(g1 / max(nseq, 1), 0.01), 0.99)
            T = min(max(num_T / (den_T + EPS), 0.01), 0.99)
            S = min(max(sk_known / (den_known + EPS), 0.01), S_CAP)
            G = min(max(cg_unknown / (den_unknown + EPS), 0.01), G_CAP)
        if best is None or tot_ll > best[0]:
            best = (tot_ll, (L0, T, S, G))
    return best[1]


def predict_seq(obs, L0, T, S, G):
    """P(correct) BEFORE each observation; sequential BKT (no leakage)."""
    pk = L0; out = np.empty(len(obs))
    for i, o in enumerate(obs):
        out[i] = pk * (1 - S) + (1 - pk) * G
        if o == 1:
            num = pk * (1 - S); den = num + (1 - pk) * G
        else:
            num = pk * S; den = num + (1 - pk) * (1 - G)
        post = num / (den + EPS)
        pk = post + (1 - post) * T
    return out


def gather_by_skill(seqs):
    d = {}
    for s in seqs:
        order = {}
        for sk, r in zip(s.skills, s.correct):
            order.setdefault(int(sk), []).append(int(r))
        for sk, rs in order.items():
            d.setdefault(sk, []).append(np.array(rs, int))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(REGISTRY))
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_root", default="preds")
    a = ap.parse_args()
    cfg = REGISTRY[a.dataset]; folds = load_folds(cfg, a.data_dir)
    allrows = []
    for fd in folds:
        train_by_skill = gather_by_skill(fd["train"])
        params = {sk: fit_skill(seqs) for sk, seqs in train_by_skill.items()}
        default = (0.3, 0.1, 0.1, 0.2)            # for skills unseen in train
        # predict test students in global step order, per-skill state carried
        rows = []
        for s in fd["test"]:
            pk = {}  # skill -> current P(known)
            for step, (sk, r) in enumerate(zip(s.skills, s.correct)):
                sk = int(sk); L0, T, S, G = params.get(sk, default)
                if sk not in pk:
                    pk[sk] = L0
                p = pk[sk] * (1 - S) + (1 - pk[sk]) * G
                rows.append((s.student, step, sk, int(r), float(p)))
                if r == 1:
                    num = pk[sk] * (1 - S); den = num + (1 - pk[sk]) * G
                else:
                    num = pk[sk] * S; den = num + (1 - pk[sk]) * (1 - G)
                post = num / (den + EPS); pk[sk] = post + (1 - post) * T
        df = pd.DataFrame(rows, columns=["student", "step", "skill", "y_true", "y_pred"])
        df["fold"] = fd["fold"]; allrows.append(df)
        m = df["step"] >= 1
        print(f"[bkt:{a.dataset}] fold {fd['fold']}: test AUC(step>=1)="
              f"{roc_auc_score(df.loc[m,'y_true'], df.loc[m,'y_pred']):.4f}  "
              f"({len(df)} rows, {len(params)} skills fit)")
    df = pd.concat(allrows, ignore_index=True)
    agg = df.groupby(["student", "step", "skill", "y_true"], as_index=False)["y_pred"].mean()
    agg.insert(0, "dataset", a.dataset)
    od = os.path.join(a.out_root, a.dataset); os.makedirs(od, exist_ok=True)
    agg.to_csv(os.path.join(od, "preds_bkt.csv"), index=False)
    m = agg["step"] >= 1
    auc = roc_auc_score(agg.loc[m, "y_true"], agg.loc[m, "y_pred"])
    print(f"[bkt:{a.dataset}] POOLED test AUC(step>=1) = {auc:.4f}  -> {od}/preds_bkt.csv")
    if auc < 0.55:
        print(f"[bkt:{a.dataset}] !! still low; inspect a few skills' fitted params.")


if __name__ == "__main__":
    # ---- self-test on synthetic BKT data (ground truth known) ----
    rng = np.random.default_rng(0)
    trueP = (0.3, 0.15, 0.12, 0.18)   # L0,T,S,G
    L0, T, S, G = trueP

    def gen(n):
        seqs = []
        for _ in range(n):
            known = rng.random() < L0; o = []
            for _ in range(rng.integers(5, 20)):
                p_corr = (1 - S) if known else G
                o.append(int(rng.random() < p_corr))
                if not known and rng.random() < T:
                    known = True
            seqs.append(np.array(o, int))
        return seqs

    train = gen(800); test = gen(300)
    est = fit_skill(train, iters=30, restarts=3)
    print("self-test  true (L0,T,S,G)=%s\n           est        =(%.3f,%.3f,%.3f,%.3f)"
          % (trueP, *est))
    ys, ps = [], []
    for o in test:
        pr = predict_seq(o, *est); ys += o.tolist(); ps += pr.tolist()
    print("self-test  test AUC = %.4f  (must be >> 0.5, slip/guess capped <0.5)"
          % roc_auc_score(ys, ps))
