import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import SEED, condition_stats, get_01_tensors, get_tensors, shift01
from models import MLP
from train_utils import (RESULTS_DIR, augment_crop_flip, count_params,
                         evaluate_classifier, load_json, save_json,
                         train_classifier)

EPOCHS = 30
SHIFT_GRID = [0, 2, 4]
JOBS = [
    ("mlp_s", (256, 128), False),
    ("mlp_m", (512, 256), False),
    ("mlp_l", (1024, 512), False),
    ("mlp_m_aug", (512, 256), True),
]


def ckpt_done(name):
    path = os.path.join(RESULTS_DIR, f"ckpt_{name}.pt")
    if not os.path.exists(path):
        return False
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["epoch"] >= EPOCHS


def load_model(name):
    model = MLP(hidden=dict((n, h) for n, h, _ in JOBS)[name])
    ck = torch.load(os.path.join(RESULTS_DIR, f"ckpt_{name}.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    return model


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    xtr, ytr = get_tensors("clean", "train")
    xva, yva = get_tensors("clean", "val")
    xte, yte = get_tensors("clean", "test")

    params = load_json(os.path.join(RESULTS_DIR, "param_counts.json")) or {}
    for name, hidden, aug in JOBS:
        params[name] = count_params(MLP(hidden=hidden))
        if not ckpt_done(name):
            print(f"training {name}", flush=True)
            model = MLP(hidden=hidden)
            train_classifier(model, xtr, ytr, xva, yva, EPOCHS, name,
                             augment_fn=augment_crop_flip if aug else None)
    save_json(os.path.join(RESULTS_DIR, "param_counts.json"), params)

    test_results = {}
    for name, _, _ in JOBS:
        model = load_model(name)
        stats, preds, confs = evaluate_classifier(model, xte, yte)
        test_results[name] = stats
        print(f"test {name}: acc={stats['acc']:.4f}", flush=True)
    save_json(os.path.join(RESULTS_DIR, "q1a_test.json"), test_results)

    xva01, _ = get_01_tensors("clean", "val")
    mean, std = condition_stats("clean")
    mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    probe = {}
    for name, _, _ in JOBS:
        model = load_model(name)
        probe[name] = {}
        for s in SHIFT_GRID:
            xs = shift01(xva01, s)
            xs = (xs - mean) / std
            stats, _, _ = evaluate_classifier(model, xs, yva)
            probe[name][str(s)] = stats["acc"]
            print(f"probe {name} shift={s}: acc={stats['acc']:.4f}", flush=True)
    save_json(os.path.join(RESULTS_DIR, "q1a_shift_probe.json"), probe)
    print("DONE q1a", flush=True)


if __name__ == "__main__":
    main()
