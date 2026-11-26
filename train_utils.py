import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

SEED = 26006

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
LOGS_DIR = os.path.join(ROOT, "logs")


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def confusion_matrix(preds, y, k=10):
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y.numpy(), preds.numpy()), 1)
    return cm


def evaluate_classifier(model, x, y, batch_size=2000):
    model.eval()
    preds, confs = [], []
    loss_sum = 0.0
    with torch.no_grad():
        for i in range(0, x.size(0), batch_size):
            xb, yb = x[i:i + batch_size], y[i:i + batch_size]
            logits = model(xb)
            loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            preds.append(pred)
            confs.append(conf)
    preds = torch.cat(preds)
    confs = torch.cat(confs)
    correct = preds == y
    n = x.size(0)
    stats = {
        "acc": (correct.float().mean().item()),
        "loss": loss_sum / n,
        "mean_conf_correct": confs[correct].mean().item() if correct.any() else 0.0,
        "mean_conf_incorrect": confs[~correct].mean().item() if (~correct).any() else 0.0,
    }
    return stats, preds, confs


def train_subset_accuracies(model, x, y, masks, batch_size=2000):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch_size):
            preds.append(model(x[i:i + batch_size]).argmax(1))
    preds = torch.cat(preds)
    return [(preds[m] == y[m]).float().mean().item() for m in masks]


def augment_crop_flip(xb, gen):
    n = xb.size(0)
    padded = F.pad(xb, (4, 4, 4, 4))
    ii = torch.randint(0, 9, (n,), generator=gen)
    jj = torch.randint(0, 9, (n,), generator=gen)
    out = torch.empty_like(xb)
    for k in range(n):
        out[k] = padded[k, :, ii[k]:ii[k] + 32, jj[k]:jj[k] + 32]
    flip = torch.rand(n, generator=gen) < 0.5
    out[flip] = torch.flip(out[flip], dims=[3])
    return out


def train_classifier(model, xtr, ytr, xva, yva, epochs, name, batch_size=128, lr=1e-3,
                     augment_fn=None, subset_masks=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    ckpt_path = os.path.join(RESULTS_DIR, f"ckpt_{name}.pt")
    hist_path = os.path.join(RESULTS_DIR, f"history_{name}.json")
    log_path = os.path.join(LOGS_DIR, f"train_{name}.log")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(SEED)
    start, history = 0, []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ck["epoch"] >= epochs:
            return ck["history"]
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        gen.set_state(ck["gen"])
        start = ck["epoch"]
        history = ck["history"]
    n = xtr.size(0)
    for epoch in range(start, epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, generator=gen)
        run_loss, run_correct = 0.0, 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = xtr[idx], ytr[idx]
            if augment_fn is not None:
                xb = augment_fn(xb, gen)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += loss.item() * xb.size(0)
            run_correct += (logits.argmax(1) == yb).sum().item()
        val_stats, _, _ = evaluate_classifier(model, xva, yva)
        entry = {
            "epoch": epoch + 1,
            "train_loss": run_loss / n,
            "train_acc": run_correct / n,
            "val_acc": val_stats["acc"],
            "val_loss": val_stats["loss"],
            "seconds": round(time.time() - t0, 1),
        }
        if subset_masks is not None:
            sub = train_subset_accuracies(model, xtr, ytr, subset_masks)
            entry["train_acc_clean_subset"] = sub[0]
            entry["train_acc_noisy_subset"] = sub[1]
        history.append(entry)
        tmp = ckpt_path + ".tmp"
        torch.save({"epoch": epoch + 1, "model": model.state_dict(), "opt": opt.state_dict(),
                    "gen": gen.get_state(), "history": history}, tmp)
        os.replace(tmp, ckpt_path)
        save_json(hist_path, history)
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
    return history


def evaluate_ae(model, x01, batch_size=2000):
    model.eval()
    mse_sum = 0.0
    n = x01.size(0)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = x01[i:i + batch_size]
            out = model(xb)
            mse_sum += F.mse_loss(out, xb, reduction="sum").item()
    mse = mse_sum / (n * 3 * 32 * 32)
    psnr = float(-10.0 * np.log10(mse))
    return mse, psnr


def train_ae(model, xtr, xva, epochs, name, batch_size=128, lr=1e-3, corrupt=False):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    ckpt_path = os.path.join(RESULTS_DIR, f"ckpt_{name}.pt")
    hist_path = os.path.join(RESULTS_DIR, f"history_{name}.json")
    log_path = os.path.join(LOGS_DIR, f"train_{name}.log")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(SEED)
    start, history = 0, []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ck["epoch"] >= epochs:
            return ck["history"]
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        gen.set_state(ck["gen"])
        start = ck["epoch"]
        history = ck["history"]
    n = xtr.size(0)
    for epoch in range(start, epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, generator=gen)
        run_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            target = xtr[idx]
            inp = corrupt01(target, gen) if corrupt else target
            out = model(inp)
            loss = F.mse_loss(out, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += loss.item() * target.size(0)
        val_mse, val_psnr = evaluate_ae(model, xva)
        entry = {
            "epoch": epoch + 1,
            "train_mse": run_loss / n,
            "val_mse": val_mse,
            "val_psnr": val_psnr,
            "seconds": round(time.time() - t0, 1),
        }
        history.append(entry)
        tmp = ckpt_path + ".tmp"
        torch.save({"epoch": epoch + 1, "model": model.state_dict(), "opt": opt.state_dict(),
                    "gen": gen.get_state(), "history": history}, tmp)
        os.replace(tmp, ckpt_path)
        save_json(hist_path, history)
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
    return history
