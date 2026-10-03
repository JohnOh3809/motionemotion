"""Train the emotion-from-motion model.

Basic run:
    python train.py --poses data/poses --labels data/labels

Options:
  --merge5        Train five classes by mapping disgusted to angry and
                  surprised to fearful. Compare results on the target data.
  --va-weight 0.3 Add an auxiliary valence/arousal loss during training.
                  The auxiliary head is excluded from the saved classifier.
  --model transformer   Use the transformer instead of the default BiLSTM.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import EMOTIONS, NUM_CLASSES, build_dataset
from model import build_model

# Optional label mapping for five-class training.
MERGE5 = {"disgusted": "angry", "surprised": "fearful"}

# Approximate valence/arousal targets in [-1, 1] for the auxiliary loss.
VA_ANCHORS = {
    "happy": (0.8, 0.6), "sad": (-0.7, -0.5), "angry": (-0.6, 0.8),
    "fearful": (-0.7, 0.7), "surprised": (0.3, 0.8), "disgusted": (-0.6, 0.3),
    "neutral": (0.0, 0.0),
}


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
    ap.add_argument("--merge5", action="store_true",
                    help="train 5 classes: disgusted->angry, surprised->fearful")
    ap.add_argument("--va-weight", type=float, default=0.0,
                    help="weight of auxiliary valence/arousal loss (0 = off)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, y = build_dataset(args.poses, args.labels, args.window, args.stride)

    if args.merge5:
        classes = [e for e in EMOTIONS if e not in MERGE5]
        remap = np.array([classes.index(MERGE5.get(e, e)) for e in EMOTIONS])
        y = remap[y]
        print(f"merged classes -> {classes}")
    else:
        classes = list(EMOTIONS)
    n_classes = len(classes)

    # 85/15 split, fixed seed so runs are comparable
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    n_val = max(1, int(0.15 * len(X)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    tr = TensorDataset(torch.from_numpy(X[tr_idx]), torch.from_numpy(y[tr_idx]))
    va = TensorDataset(torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx]))
    tr_dl = DataLoader(tr, batch_size=args.batch, shuffle=True)
    va_dl = DataLoader(va, batch_size=args.batch)

    # Weight classes by inverse frequency to reduce majority-class bias.
    counts = np.bincount(y[tr_idx], minlength=n_classes).astype(np.float32)
    weights = torch.tensor(counts.sum() / np.clip(counts, 1, None) / n_classes,
                           dtype=torch.float32, device=device)

    model = build_model(args.model, num_classes=n_classes).to(device)

    # Optional valence/arousal head uses the shared encoding.
    va_head, va_targets = None, None
    params = list(model.parameters())
    if args.va_weight > 0:
        va_head = nn.Linear(model.pooled_dim, 2).to(device)
        va_targets = torch.tensor([VA_ANCHORS[c] for c in classes],
                                  dtype=torch.float32, device=device)
        params += list(va_head.parameters())
        print(f"auxiliary valence/arousal head on (weight {args.va_weight})")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = nn.CrossEntropyLoss(weight=weights)
    mse = nn.MSELoss()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pooled = model.pooled(xb)
            loss = crit(model.head(pooled), yb)
            if va_head is not None:
                loss = loss + args.va_weight * mse(va_head(pooled), va_targets[yb])
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(xb)
        sched.step()

        model.eval()
        correct, total = 0, 0
        conf = np.zeros((n_classes, n_classes), dtype=int)
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
            # save the class list too — infer.py and the app read it from here
            torch.save({"model": model.state_dict(), "arch": args.model,
                        "window": args.window, "emotions": classes},
                       out_dir / "best.pt")

    print(f"\nbest val acc: {best_acc:.3f} — saved to {out_dir/'best.pt'}")
    print("val confusion matrix (rows=true, cols=pred):")
    print("        " + " ".join(f"{e[:6]:>6}" for e in classes))
    for i, row in enumerate(conf):
        print(f"{classes[i][:7]:>7} " + " ".join(f"{v:6d}" for v in row))


if __name__ == "__main__":
    main()
