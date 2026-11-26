import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import SEED, get_tensors
from models import CNN
from train_utils import (RESULTS_DIR, confusion_matrix, count_params,
                         evaluate_classifier, load_json, save_json,
                         train_classifier)

EPOCHS = 30


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    xtr, ytr = get_tensors("clean", "train")
    xva, yva = get_tensors("clean", "val")
    xte, yte = get_tensors("clean", "test")

    model = CNN()
    params = load_json(os.path.join(RESULTS_DIR, "param_counts.json")) or {}
    params["cnn_c"] = count_params(model)
    save_json(os.path.join(RESULTS_DIR, "param_counts.json"), params)
    print(f"params {params['cnn_c']}", flush=True)

    ckpt = os.path.join(RESULTS_DIR, "ckpt_cnn_c.pt")
    done = False
    if os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        done = ck["epoch"] >= EPOCHS
    if not done:
        train_classifier(model, xtr, ytr, xva, yva, EPOCHS, "cnn_c")

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    stats, preds, confs = evaluate_classifier(model, xte, yte)
    cm = confusion_matrix(preds, yte)
    result = {
        "cnn_c": stats,
        "confusion_matrix": cm.tolist(),
        "best_val_acc": max(e["val_acc"] for e in ck["history"]),
        "final_val_acc": ck["history"][-1]["val_acc"],
    }
    save_json(os.path.join(RESULTS_DIR, "q2a_results.json"), result)
    print(f"cnn_c test acc {stats['acc']:.4f} conf_correct {stats['mean_conf_correct']:.3f} conf_wrong {stats['mean_conf_incorrect']:.3f}", flush=True)
    print("DONE q2a", flush=True)


if __name__ == "__main__":
    main()
