"""Train the emotion-from-motion model.

Usage:
    python train.py --poses data/poses --labels data/labels \
        [--model lstm|transformer] [--window 45] [--epochs 40] [--out checkpoints]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import EMOTIONS, NUM_CLASSES, build_dataset
from model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--model", default="lstm", choices=["lstm", "transformer"])
    ap.add_argument("--window", type=int, default=45)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, y = build_dataset(args.poses, args.labels, args.window, args.stride)

    # split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    n_val = max(1, int(0.15 * len(X)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    tr = TensorDataset(torch.from_numpy(X[tr_idx]), torch.from_numpy(y[tr_idx]))
    va = TensorDataset(torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx]))
    tr_dl = DataLoader(tr, batch_size=args.batch, shuffle=True)
    va_dl = DataLoader(va, batch_size=args.batch)

    # class weights for imbalance (face labels are usually neutral-heavy)
    counts = np.bincount(y[tr_idx], minlength=NUM_CLASSES).astype(np.float32)
    weights = torch.tensor(counts.sum() / np.clip(counts, 1, None) / NUM_CLASSES,
                           dtype=torch.float32, device=device)

    model = build_model(args.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = nn.CrossEntropyLoss(weight=weights)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(xb)
        sched.step()

        model.eval()
        correct, total = 0, 0
        conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        with torch.no_grad():
            for xb, yb in va_dl:
                pred = model(xb.to(device)).argmax(1).cpu()
                correct += (pred == yb).sum().item()
                total += len(yb)
                for t, p in zip(yb.numpy(), pred.numpy()):
                    conf[t, p] += 1
        acc = correct / max(total, 1)
        print(f"epoch {epoch:3d} | train loss {tr_loss/len(tr):.4f} | val acc {acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "arch": args.model,
                        "window": args.window, "emotions": EMOTIONS},
                       out_dir / "best.pt")

    print(f"\nbest val acc: {best_acc:.3f} — saved to {out_dir/'best.pt'}")
    print("val confusion matrix (rows=true, cols=pred):")
    print("        " + " ".join(f"{e[:6]:>6}" for e in EMOTIONS))
    for i, row in enumerate(conf):
        print(f"{EMOTIONS[i][:7]:>7} " + " ".join(f"{v:6d}" for v in row))


if __name__ == "__main__":
    main()
