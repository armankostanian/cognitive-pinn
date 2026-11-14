"""
Cognitive-PINN loader over the CLEAN canonical data, producing the SAME bundle
format as your cpinn_assist09_loader.load_assist2009, so you re-run CPINN on
clean data with a one-line change in your notebook.

Reuses your repo's student_to_cpinn_format / StudentRecord / ASSISTments2009Bundle
(import path = your repo's src/). Uses the SAME folds.csv and SAME (student,step)
keys as the baselines (datasets.py) so predictions align for significance.

Usage in your training notebook (replace load_assist2009):

    import sys; sys.path.insert(0, "/path/to/cognitive-pinn/src")
    from cpinn_clean_loader import load_clean
    bundle = load_clean("assist2009", fold=fold, clean_root="clean")
    # bundle.train / bundle.val / bundle.test / bundle.K  -- as before
    # ... train model ...
    from cpinn_dump_predictions import dump_cpinn_fold
    dump_cpinn_fold(model, bundle.test, "assist2009", fold, "preds", DEVICE)

After all folds: aggregate (cpinn_dump_predictions.py --aggregate).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

# from YOUR repo (src/ on sys.path):
from cpinn_assist09_loader import (StudentRecord, student_to_cpinn_format,
                                   ASSISTments2009Bundle)
from datasets import val_split   # shared deterministic val split


def load_clean(dataset, fold, clean_root="clean"):
    """fold is 0-based here (0..K_FOLDS-1) to match folds.csv."""
    data = pd.read_csv(os.path.join(clean_root, dataset, "data.csv"))
    folds = pd.read_csv(os.path.join(clean_root, dataset, "folds.csv"))
    K = int(data["skill"].max()) + 1
    fold_of = dict(zip(folds["student"].astype(int), folds["fold"].astype(int)))

    seqs = {}
    for st, g in data.sort_values(["student", "step"]).groupby("student", sort=True):
        seqs[int(st)] = (g["skill"].to_numpy(np.int64), g["correct"].to_numpy(np.int64))

    test_ids = [s for s in seqs if fold_of.get(s, -1) == fold]
    nontest = [s for s in seqs if fold_of.get(s, -1) != fold]
    val_ids = val_split(nontest)
    train_ids = [s for s in nontest if s not in val_ids]

    def to_records(ids):
        return [StudentRecord(skill_seq=seqs[s][0], response_seq=seqs[s][1]) for s in ids]

    def to_bundle_list(ids):
        # skills are already dense 0..K-1 -> pass straight to your converter
        return [student_to_cpinn_format(r, K) for r in to_records(ids)]

    return ASSISTments2009Bundle(
        train=to_bundle_list(train_ids),
        val=to_bundle_list(sorted(val_ids)),
        test=to_bundle_list(test_ids),
        concept_index={i: i for i in range(K)},
        K=K,
        skill_names={},
    )
