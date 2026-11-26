import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import SEED, get_tensors
from models import CNN, MLP
from train_utils import (RESULTS_DIR, confusion_matrix, count_params,
                         evaluate_classifier, save_json, train_classifier)

EPOCHS = 30


def train_and_eval(name, model, condition):
    xtr, ytr = get_tensors(condition, "train")
    xva, yva = get_tensors(condition, "val")
    xte, yte = get_tensors(condition, "test")
    ckpt = os.path.join(RESULTS_DIR, f"ckpt_{name}.pt")
    done = False
    if os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        done = ck["epoch"] >= EPOCHS
    if not done:
        train_classifier(model, xtr, ytr, xva, yva, EPOCHS, name)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    stats, preds, confs = evaluate_classifier(model, xte, yte)
    return {
        "stats": stats,
        "confusion_matrix": confusion_matrix(preds, yte).tolist(),
        "best_val_acc": max(e["val_acc"] for e in ck["history"]),
        "final_val_acc": ck["history"][-1]["val_acc"],
        "params": count_params(model),
    }


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    result = {}
    for name, model in [("mlp_m_perm", MLP(hidden=(512, 256))), ("cnn_c_perm", CNN())]:
        print(f"training {name}", flush=True)
        result[name] = train_and_eval(name, model, "permuted")
        print(f"{name} test acc {result[name]['stats']['acc']:.4f}", flush=True)
    save_json(os.path.join(RESULTS_DIR, "q2b_results.json"), result)
    print("DONE q2b", flush=True)


if __name__ == "__main__":
    main()
