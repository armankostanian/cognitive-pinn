"""
ASSISTments-2009 data loader for Cognitive-PINN.

Input file format (DKVMN-style preprocessed CSV):
    line 3*i + 0: N           -- length of student i's history
    line 3*i + 1: c1,c2,...,cN -- skill (concept) ids
    line 3*i + 2: r1,r2,...,rN -- binary responses (0 or 1)

We re-interpret interaction index as pseudo-time:
    t_i = i * dt,   dt = 1 (in arbitrary "interaction-time" units).
This standard choice in DKT/DKVMN/SAKT papers is justified because
ASSISTments-2009 does not provide reliable wall-clock timestamps.

For the Cognitive-PINN ODE we need:
    * t_grid : sequence of times for ODE integration
    * k0     : initial knowledge state (one-hot or zeros)
    * practice_fn(t) : practice intensity vector at time t
    * item_times, item_idx, q_vecs, y_true: as in the synthetic loader.

Each interaction simultaneously practises the relevant concept AND
queries it. We model this by setting practice intensity to a delta-like
pulse around interaction time, and by treating the same skill_id as
both the practised concept and the queried one (single-skill items).
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch


# -----------------------------------------------------------------------------
# 1. Raw parsing
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class StudentRecord:
    """A single student's interaction sequence in raw form."""
    skill_seq: np.ndarray   # (N,) skill ids, 1-indexed in the source files
    response_seq: np.ndarray  # (N,) 0/1 responses


def parse_dkvmn_csv(path: str | Path,
                    min_length: int = 3,
                    max_length: Optional[int] = None) -> List[StudentRecord]:
    """Parse a DKVMN-style preprocessed CSV file."""
    path = Path(path)
    students: List[StudentRecord] = []
    with open(path, "r") as f:
        lines = [ln.rstrip("\n").rstrip(",") for ln in f if ln.strip()]
    i = 0
    while i + 2 < len(lines):
        try:
            N = int(lines[i].strip())
        except ValueError:
            i += 1
            continue
        if N <= 0 or N != len(lines[i + 1].split(",")):
            i += 1
            continue
        skills = np.array([int(x) for x in lines[i + 1].split(",")],
                           dtype=np.int64)
        responses = np.array([int(x) for x in lines[i + 2].split(",")],
                              dtype=np.int64)
        if len(skills) >= min_length and len(skills) == len(responses):
            if max_length is not None and len(skills) > max_length:
                # truncate to most recent max_length interactions
                skills = skills[-max_length:]
                responses = responses[-max_length:]
            students.append(StudentRecord(skill_seq=skills,
                                          response_seq=responses))
        i += 3
    return students


def parse_skill_mapping(path: str | Path) -> Dict[int, str]:
    """Parse skill id -> human-readable name mapping."""
    mapping: Dict[int, str] = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                try:
                    mapping[int(parts[0])] = parts[1].strip()
                except ValueError:
                    continue
    return mapping


# -----------------------------------------------------------------------------
# 2. Build a compact concept ID space
# -----------------------------------------------------------------------------

def build_concept_index(students: List[StudentRecord]) -> Dict[int, int]:
    """Map source skill ids (sparse, possibly not contiguous) to dense
    0-based indices."""
    all_ids = set()
    for s in students:
        all_ids.update(s.skill_seq.tolist())
    sorted_ids = sorted(all_ids)
    return {sid: idx for idx, sid in enumerate(sorted_ids)}


def remap(students: List[StudentRecord],
          concept_index: Dict[int, int]) -> List[StudentRecord]:
    out: List[StudentRecord] = []
    for s in students:
        new_skills = np.array([concept_index[sid] for sid in s.skill_seq],
                               dtype=np.int64)
        out.append(StudentRecord(skill_seq=new_skills,
                                 response_seq=s.response_seq))
    return out


# -----------------------------------------------------------------------------
# 3. Convert one student to Cognitive-PINN tensors
# -----------------------------------------------------------------------------

def student_to_cpinn_format(student: StudentRecord, K: int,
                            dt: float = 1.0,
                            practice_pulse_width: float = 0.5,
                            practice_amplitude: float = 1.0,
                            ) -> Dict[str, torch.Tensor]:
    """Convert one student's sequence into the dict format expected by the
    model's training loop.

    The pseudo-time grid has length N+1 (one extra point at t=0).
    Practice intensity at time t is a piece-wise constant pulse centred on
    each past interaction, decaying outside a half-width window. Item
    queries happen at the interaction times themselves.
    """
    N = len(student.skill_seq)
    skills = student.skill_seq                      # (N,) concept indices
    responses = student.response_seq                # (N,) 0/1

    # Pseudo-time
    item_times = np.arange(1, N + 1, dtype=np.float32) * dt   # (N,)
    t_grid = np.arange(0, N + 1, dtype=np.float32) * dt       # (N+1,)

    # Q-matrix: each item is identified by skill_id; one-hot row.
    # We treat skill_id == item_id == concept_id (one item per concept
    # per interaction in the DKVMN-style preprocessed data).
    q_vecs = np.zeros((N, K), dtype=np.float32)
    q_vecs[np.arange(N), skills] = 1.0

    item_idx = skills.astype(np.int64)              # (N,) item id = concept id

    # Practice schedule encoded as piece-wise constant pulses.
    # We bake the schedule into a NumPy table (T_grid x K) and create
    # a closure that interpolates on call.
    pulse_table = np.zeros((len(t_grid), K), dtype=np.float32)
    for i, (t_i, c_i) in enumerate(zip(item_times, skills)):
        # find indices in t_grid covered by the pulse [t_i - w, t_i + w]
        mask = (t_grid >= t_i - practice_pulse_width) & \
               (t_grid <= t_i + practice_pulse_width)
        pulse_table[mask, c_i] += practice_amplitude

    pulse_table_t = torch.tensor(pulse_table)       # (T, K)
    t_grid_t = torch.tensor(t_grid)                 # (T,)

    # Mutable container for the device-cached version of the pulse table.
    # We lazily move the tables to the device of the first query and keep
    # them there.
    _cache = {"pulse": pulse_table_t, "tgrid": t_grid_t, "device": "cpu"}

    def practice_fn(t: torch.Tensor) -> torch.Tensor:
        """Piece-wise constant interpolation of the pulse table; moves
        the cache to the device of t the first time."""
        target_device = t.device if hasattr(t, "device") else torch.device("cpu")
        if str(target_device) != _cache["device"]:
            _cache["pulse"] = _cache["pulse"].to(target_device)
            _cache["tgrid"] = _cache["tgrid"].to(target_device)
            _cache["device"] = str(target_device)
        idx = torch.searchsorted(_cache["tgrid"], t).clamp(0, len(_cache["tgrid"]) - 1)
        return _cache["pulse"][idx]

    return {
        "t_grid": torch.tensor(t_grid),
        "k0": torch.full((K,), 0.1, dtype=torch.float32),  # cold start
        "practice_fn": practice_fn,
        "item_times": torch.tensor(item_times),
        "item_idx": torch.tensor(item_idx, dtype=torch.long),
        "q_vecs": torch.tensor(q_vecs),
        "y_true": torch.tensor(responses, dtype=torch.float32),
    }


# -----------------------------------------------------------------------------
# 4. Top-level convenience loader
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class ASSISTments2009Bundle:
    train: List[Dict[str, torch.Tensor]]
    val: List[Dict[str, torch.Tensor]]
    test: List[Dict[str, torch.Tensor]]
    concept_index: Dict[int, int]
    K: int
    skill_names: Dict[int, str]


def load_assist2009(data_dir: str | Path,
                    fold: int = 1,
                    min_length: int = 3,
                    max_length: int = 200,
                    sample_train: Optional[int] = None,
                    sample_test: Optional[int] = None,
                    seed: int = 42) -> ASSISTments2009Bundle:
    """Load ASSISTments-2009 for a given fold (1..5) of the DKVMN split.

    Args:
        data_dir: path to the directory with assist2009_updated_*.csv files
        fold: 1..5 for the canonical DKVMN cross-validation split
        sample_train, sample_test: sub-sample students for fast prototyping
    """
    data_dir = Path(data_dir)
    rng = np.random.default_rng(seed)

    train_path = data_dir / f"assist2009_updated_train{fold}.csv"
    valid_path = data_dir / f"assist2009_updated_valid{fold}.csv"
    test_path = data_dir / "assist2009_updated_test.csv"
    skill_path = data_dir / "assist2009_updated_skill_mapping.txt"

    print(f"  parsing {train_path.name}...")
    train_raw = parse_dkvmn_csv(train_path, min_length, max_length)
    print(f"  parsing {valid_path.name}...")
    valid_raw = parse_dkvmn_csv(valid_path, min_length, max_length)
    print(f"  parsing {test_path.name}...")
    test_raw = parse_dkvmn_csv(test_path, min_length, max_length)

    if sample_train is not None and len(train_raw) > sample_train:
        idx = rng.choice(len(train_raw), sample_train, replace=False)
        train_raw = [train_raw[i] for i in idx]
    if sample_test is not None and len(valid_raw) > sample_test:
        idx = rng.choice(len(valid_raw), sample_test, replace=False)
        valid_raw = [valid_raw[i] for i in idx]
    if sample_test is not None and len(test_raw) > sample_test:
        idx = rng.choice(len(test_raw), sample_test, replace=False)
        test_raw = [test_raw[i] for i in idx]

    print(f"  building concept index from train + val + test...")
    concept_index = build_concept_index(train_raw + valid_raw + test_raw)
    K = len(concept_index)
    print(f"  K = {K} concepts")

    train_raw = remap(train_raw, concept_index)
    valid_raw = remap(valid_raw, concept_index)
    test_raw = remap(test_raw, concept_index)

    print(f"  converting to Cognitive-PINN tensor format...")
    train = [student_to_cpinn_format(s, K) for s in train_raw]
    val = [student_to_cpinn_format(s, K) for s in valid_raw]
    test = [student_to_cpinn_format(s, K) for s in test_raw]

    skill_names = {}
    if skill_path.exists():
        skill_names = parse_skill_mapping(skill_path)

    return ASSISTments2009Bundle(
        train=train, val=val, test=test,
        concept_index=concept_index, K=K,
        skill_names=skill_names,
    )


# -----------------------------------------------------------------------------
# 5. Sanity check
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/claude/data/DKVMN/data/assist2009_updated"

    print(f"[Loading ASSISTments-2009 from {path}]")
    bundle = load_assist2009(path, fold=1,
                              sample_train=20, sample_test=10)

    print()
    print(f"Bundle stats:")
    print(f"  K = {bundle.K} concepts")
    print(f"  train: {len(bundle.train)} students")
    print(f"  val:   {len(bundle.val)} students")
    print(f"  test:  {len(bundle.test)} students")
    if bundle.train:
        s = bundle.train[0]
        print()
        print(f"First student tensors:")
        print(f"  t_grid:     {s['t_grid'].shape} dtype={s['t_grid'].dtype}")
        print(f"  k0:         {s['k0'].shape}")
        print(f"  item_times: {s['item_times'].shape}")
        print(f"  item_idx:   {s['item_idx'].shape}, range [{s['item_idx'].min()},{s['item_idx'].max()}]")
        print(f"  q_vecs:     {s['q_vecs'].shape}, sum per row = {s['q_vecs'].sum(dim=1)[:5].tolist()}")
        print(f"  y_true:     {s['y_true'].shape}, mean = {float(s['y_true'].mean()):.3f}")
        # quick test of practice_fn
        t_test = s['t_grid'][5]
        p = s['practice_fn'](t_test)
        print(f"  practice_fn at t={float(t_test):.1f}: nonzero in {(p > 0).sum().item()} concepts")

    print()
    print("Sanity check passed.")
