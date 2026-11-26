import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10

SEED = 26006
NOISE_FRACTION = 0.20
NOISE_SIGMA = 0.15
SVM_SUBSET_SIZE = 10000

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
SPLIT_PATH = os.path.join(DATA_DIR, "split_indices.npz")
PERM_PATH = os.path.join(DATA_DIR, "permutation.npz")
NOISE_PATH = os.path.join(DATA_DIR, "noise_indices.npz")


def raw_uint8():
    dtrain = CIFAR10(root=DATA_DIR, train=True, download=False)
    dtest = CIFAR10(root=DATA_DIR, train=False, download=False)
    return dtrain.data, np.array(dtrain.targets), dtest.data, np.array(dtest.targets)


def build_split():
    if not os.path.exists(SPLIT_PATH):
        _, y50, _, _ = raw_uint8()
        idx = np.arange(len(y50))
        train_idx, val_idx = train_test_split(idx, test_size=5000, stratify=y50, random_state=SEED)
        np.savez(SPLIT_PATH, train_idx=train_idx, val_idx=val_idx)
    split = np.load(SPLIT_PATH)
    return split["train_idx"], split["val_idx"]


def build_permutation():
    if not os.path.exists(PERM_PATH):
        rng = np.random.default_rng(SEED + 1)
        np.savez(PERM_PATH, perm=rng.permutation(3072))
    return np.load(PERM_PATH)["perm"]


def build_noise():
    if not os.path.exists(NOISE_PATH):
        train_idx, _ = build_split()
        _, y50, _, _ = raw_uint8()
        ytr = y50[train_idx]
        rng = np.random.default_rng(SEED + 2)
        n_noisy = int(round(NOISE_FRACTION * len(ytr)))
        noisy_pos = rng.choice(len(ytr), size=n_noisy, replace=False)
        shift = rng.integers(1, 10, size=n_noisy)
        noisy_y = ytr.copy()
        noisy_y[noisy_pos] = (ytr[noisy_pos] + shift) % 10
        np.savez(NOISE_PATH, noisy_pos=noisy_pos, noisy_y=noisy_y)
    noise = np.load(NOISE_PATH)
    return noise["noisy_y"], noise["noisy_pos"]


def _apply_condition(x_hwc, condition):
    if condition == "clean":
        return x_hwc
    if condition == "greyscale":
        lum = 0.299 * x_hwc[..., 0] + 0.587 * x_hwc[..., 1] + 0.114 * x_hwc[..., 2]
        return np.repeat(lum[..., None], 3, axis=3)
    if condition == "permuted":
        perm = build_permutation()
        n = x_hwc.shape[0]
        chw = x_hwc.transpose(0, 3, 1, 2).reshape(n, -1)
        return chw[:, perm].reshape(n, 3, 32, 32).transpose(0, 2, 3, 1)
    raise ValueError(condition)


def _chunk_condition_iter(x_u8, condition, chunk=5000):
    for i in range(0, x_u8.shape[0], chunk):
        xc = x_u8[i:i + chunk].astype(np.float32) / 255.0
        yield _apply_condition(xc, condition)


def condition_stats(condition):
    stats_path = os.path.join(DATA_DIR, f"stats_{condition}.npz")
    if not os.path.exists(stats_path):
        x50, _, _, _ = raw_uint8()
        train_idx, _ = build_split()
        x_u8 = x50[train_idx]
        n_pix = x_u8.shape[0] * 32 * 32
        total = np.zeros(3, dtype=np.float64)
        total_sq = np.zeros(3, dtype=np.float64)
        for xc in _chunk_condition_iter(x_u8, condition):
            v = xc.reshape(-1, 3).astype(np.float64)
            total += v.sum(axis=0)
            total_sq += np.square(v).sum(axis=0)
        mean = total / n_pix
        std = np.sqrt(np.maximum(total_sq / n_pix - mean ** 2, 0.0))
        np.savez(stats_path, mean=mean, std=std)
    stats = np.load(stats_path)
    return stats["mean"], stats["std"]


def get_01_tensors(condition, split):
    x50, y50, xte, yte = raw_uint8()
    train_idx, val_idx = build_split()
    if split == "train":
        x_u8, y = x50[train_idx], y50[train_idx]
    elif split == "val":
        x_u8, y = x50[val_idx], y50[val_idx]
    elif split == "test":
        x_u8, y = xte, yte
    else:
        raise ValueError(split)
    n = x_u8.shape[0]
    out = torch.empty((n, 3, 32, 32), dtype=torch.float32)
    pos = 0
    for xc in _chunk_condition_iter(x_u8, condition):
        k = xc.shape[0]
        out[pos:pos + k] = torch.from_numpy(xc.transpose(0, 3, 1, 2).copy())
        pos += k
    return out, torch.from_numpy(y.astype(np.int64))


def get_tensors(condition, split):
    x, y = get_01_tensors(condition, split)
    mean, std = condition_stats(condition)
    mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return (x - mean) / std, y


def normalize01(x01, condition):
    mean, std = condition_stats(condition)
    mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return (x01 - mean) / std


def shift01(x01, s):
    if s == 0:
        return x01
    padded = F.pad(x01, (s, s, s, s))
    return padded[:, :, :32, :32]


def corrupt01(x01, gen):
    noise = torch.randn(x01.shape, generator=gen)
    return (x01 + NOISE_SIGMA * noise).clamp(0.0, 1.0)
