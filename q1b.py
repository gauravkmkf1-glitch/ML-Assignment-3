import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import SEED, build_noise, get_tensors
from models import MLP
from train_utils import count_params, evaluate_classifier, save_json, train_classifier

EPOCHS = 25


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    xtr, ytr_clean = get_tensors("clean", "train")
    xva, yva = get_tensors("clean", "val")

    noisy_y, noisy_pos = build_noise()
    ytr_noisy = torch.from_numpy(noisy_y.astype(np.int64))
    noisy_mask = torch.zeros(len(ytr_clean), dtype=torch.bool)
    noisy_mask[torch.from_numpy(noisy_pos.astype(np.int64))] = True
    clean_mask = ~noisy_mask
    print(f"clean subset {int(clean_mask.sum())} noisy subset {int(noisy_mask.sum())}", flush=True)

    model = MLP(hidden=(1024, 512))
    print(f"params {count_params(model)}", flush=True)
    train_classifier(model, xtr, ytr_noisy, xva, yva, EPOCHS, "mlp_l_noise",
                     subset_masks=(clean_mask, noisy_mask))
    print("DONE q1b", flush=True)


if __name__ == "__main__":
    main()
