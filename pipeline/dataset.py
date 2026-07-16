"""Build windowed (sequence, label) training samples from pose + label files.

Pairs data/poses/<name>.npz with data/labels/<name>_labels.npz, slices the
pose stream into fixed-length windows, and assigns each window the majority
confident face label. Windows without a confident majority are dropped.

Normalization (per frame): center on hip midpoint, scale by torso length —
this removes camera distance/position so the model sees only body shape+motion.
Feature per frame = normalized xy (33*2) + xy velocities (33*2) = 132 dims.
"""
from pathlib import Path

import numpy as np

EMOTIONS = ["happy", "sad", "angry", "fearful", "surprised", "disgusted", "neutral"]
NUM_CLASSES = len(EMOTIONS)
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24
FEAT_DIM = 33 * 2 * 2  # xy + velocity xy


def normalize_frames(kp: np.ndarray) -> np.ndarray:
    """(T, 33, 4) -> (T, 33, 2) centered/scaled xy."""
    xy = kp[:, :, :2].copy()
    hip_mid = (xy[:, L_HIP] + xy[:, R_HIP]) / 2          # (T, 2)
    sh_mid = (xy[:, L_SHOULDER] + xy[:, R_SHOULDER]) / 2
    torso = np.linalg.norm(sh_mid - hip_mid, axis=1, keepdims=True)  # (T, 1)
    torso = np.clip(torso, 1e-4, None)
    return (xy - hip_mid[:, None, :]) / torso[:, None, :]


def to_features(kp: np.ndarray) -> np.ndarray:
    """(T, 33, 4) -> (T, FEAT_DIM) position + velocity features."""
    xy = normalize_frames(kp)                     # (T, 33, 2)
    vel = np.zeros_like(xy)
    vel[1:] = xy[1:] - xy[:-1]
    feats = np.concatenate([xy, vel], axis=2)     # (T, 33, 4)
    return feats.reshape(len(kp), -1).astype(np.float32)


def window_label(frame_idx, labels, confs, start, end, min_frac=0.5):
    """Majority confident label within [start, end), or -1."""
    m = (frame_idx >= start) & (frame_idx < end) & (labels >= 0)
    if m.sum() == 0:
        return -1
    votes = np.bincount(labels[m], weights=confs[m], minlength=NUM_CLASSES)
    winner = int(votes.argmax())
    if (labels[m] == winner).mean() < min_frac:
        return -1  # no clear majority — ambiguous window
    return winner


def build_dataset(poses_dir, labels_dir, window=45, stride=15):
    """Returns X (N, window, FEAT_DIM), y (N,)."""
    poses_dir, labels_dir = Path(poses_dir), Path(labels_dir)
    X, y = [], []
    for pose_file in sorted(poses_dir.glob("*.npz")):
        label_file = labels_dir / (pose_file.stem + "_labels.npz")
        if not label_file.exists():
            print(f"skip {pose_file.name}: no label file")
            continue
        pd = np.load(pose_file)
        ld = np.load(label_file)
        kp = pd["keypoints"]                       # (T, 33, 4)
        if len(kp) < window:
            continue
        # drop frames where no person was detected
        valid = ~np.isnan(kp[:, 0, 0])
        kp = np.where(np.isnan(kp), 0.0, kp)
        feats = to_features(kp)

        for start in range(0, len(kp) - window + 1, stride):
            end = start + window
            if valid[start:end].mean() < 0.8:
                continue
            lbl = window_label(ld["frame_idx"], ld["label"], ld["conf"], start, end)
            if lbl < 0:
                continue
            X.append(feats[start:end])
            y.append(lbl)

    if not X:
        raise SystemExit("No labeled windows produced — collect more video or lower --min-conf in auto_label.py")
    X, y = np.stack(X), np.array(y, dtype=np.int64)
    print(f"dataset: {len(X)} windows, class counts: "
          + ", ".join(f"{EMOTIONS[i]}={c}" for i, c in enumerate(np.bincount(y, minlength=NUM_CLASSES))))
    return X, y
