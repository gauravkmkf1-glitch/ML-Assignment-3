import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build_split, build_permutation, build_noise, condition_stats


def class_counts(y):
    return np.bincount(np.asarray(y), minlength=10).tolist()


def main():
    train_idx, val_idx = build_split()
    perm = build_permutation()
    noisy_y, noisy_pos = build_noise()
    for condition in ["clean", "permuted", "greyscale"]:
        mean, std = condition_stats(condition)
        print(f"stats {condition}: mean={np.round(mean, 4).tolist()} std={np.round(std, 4).tolist()}")
    _, y50, _, yte = __import__("data").raw_uint8()
    ytr = y50[train_idx]
    yva = y50[val_idx]
    print(f"train size {len(train_idx)} per-class {class_counts(ytr)}")
    print(f"val size {len(val_idx)} per-class {class_counts(yva)}")
    print(f"test size {len(yte)} per-class {class_counts(yte)}")
    print(f"permutation first 8 {perm[:8].tolist()}")
    print(f"noisy labels {len(noisy_pos)} of {len(ytr)} fraction {len(noisy_pos) / len(ytr):.3f}")
    print("SPLIT READY")


if __name__ == "__main__":
    main()
