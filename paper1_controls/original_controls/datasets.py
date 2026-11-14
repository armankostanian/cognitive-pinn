"""
Loader over the CLEAN canonical format produced by preprocess.py.
Shared by run_dkt.py and bkt_simple.py so all baselines use the SAME folds and
the SAME (student,step) keys as Cognitive-PINN (cpinn_clean_loader.py).

clean/<dataset>/data.csv : student, step, skill, correct  (dense ints)
clean/<dataset>/folds.csv: student, fold
"""
from __future__ import annotations
import os, dataclasses
import numpy as np
import pandas as pd

VAL_FRAC = 0.125
VAL_SEED = 0


@dataclasses.dataclass
class Seq:
    student: int
    skills: np.ndarray
    correct: np.ndarray


def val_split(nontest_students, frac=VAL_FRAC, seed=VAL_SEED):
    """Deterministic val subset of the non-test students (shared by all models)."""
    s = np.array(sorted(nontest_students))
    rng = np.random.default_rng(seed); rng.shuffle(s)
    k = max(1, int(round(len(s) * frac)))
    return set(int(x) for x in s[:k])


def _read(dataset, clean_root):
    data = pd.read_csv(os.path.join(clean_root, dataset, "data.csv"))
    folds = pd.read_csv(os.path.join(clean_root, dataset, "folds.csv"))
    return data, folds


def load_folds(dataset, clean_root="clean"):
    """Returns list of folds: {fold, K, train:[Seq], val:[Seq], test:[Seq]}."""
    data, folds = _read(dataset, clean_root)
    K = int(data["skill"].max()) + 1
    fold_of = dict(zip(folds["student"].astype(int), folds["fold"].astype(int)))
    nfolds = int(folds["fold"].nunique())
    seqs = {}
    for st, g in data.sort_values(["student", "step"]).groupby("student", sort=True):
        seqs[int(st)] = (g["skill"].to_numpy(np.int64), g["correct"].to_numpy(np.int64))

    out = []
    for f in range(nfolds):
        test_ids = [s for s in seqs if fold_of.get(s, -1) == f]
        nontest = [s for s in seqs if fold_of.get(s, -1) != f]
        val_ids = val_split(nontest)
        train_ids = [s for s in nontest if s not in val_ids]
        mk = lambda ids: [Seq(s, seqs[s][0], seqs[s][1]) for s in ids]
        out.append({"fold": f + 1, "K": K, "train": mk(train_ids),
                    "val": mk(sorted(val_ids)), "test": mk(test_ids)})
    return out
