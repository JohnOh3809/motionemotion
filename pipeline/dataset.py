"""Turns pose files + label files into actual training data.

The deal: extract_poses.py gives us data/poses/<name>.npz, the labelers give
us data/labels/<name>_labels.npz. This pairs them up, chops the pose stream
into fixed windows, and gives each window whatever label won the vote inside
it. Windows where no label clearly wins get thrown out — ambiguous data is
worse than less data (learned this the hard way).

Normalization: every frame gets centered on the hip midpoint and scaled by
torso length. So it doesn't matter if you filmed from 2m or 5m away, or if
you're standing off to the side — the model only ever sees body shape and
how it moves. 33 joints * (xy + velocity xy) = 132 numbers per frame.
"""
from pathlib import Path

import numpy as np

# this order is basically sacred — the app, the labelers and the checkpoints
# all assume it. don't reorder!
EMOTIONS = ["happy", "sad", "angry", "fearful", "surprised", "disgusted", "neutral"]
NUM_CLASSES = len(EMOTIONS)
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24  # mediapipe indices
FEAT_DIM = 33 * 2 * 2


def normalize_frames(kp: np.ndarray) -> np.ndarray:
    """(T, 33, 4) -> (T, 33, 2). Center on hips, scale by torso."""
    xy = kp[:, :, :2].copy()
    hip_mid = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    sh_mid = (xy[:, L_SHOULDER] + xy[:, R_SHOULDER]) / 2
    torso = np.linalg.norm(sh_mid - hip_mid, axis=1, keepdims=True)
    torso = np.clip(torso, 1e-4, None)  # divide-by-zero guard, don't ask how i know
    return (xy - hip_mid[:, None, :]) / torso[:, None, :]


# nb: if you change to_features in ANY way, change toModelFeatures() in
# app/index.html to match, or the in-browser model will silently predict garbage.
def to_features(kp: np.ndarray) -> np.ndarray:
    """(T, 33, 4) -> (T, 132). Position + frame-to-frame velocity."""
    xy = normalize_frames(kp)
    vel = np.zeros_like(xy)
    vel[1:] = xy[1:] - xy[:-1]  # first frame just gets zero velocity, fine
    feats = np.concatenate([xy, vel], axis=2)
    return feats.reshape(len(kp), -1).astype(np.float32)


def window_label(frame_idx, labels, confs, start, end, min_frac=0.5):
    """Confidence-weighted vote inside [start, end). Returns -1 if it's a mess."""
    m = (frame_idx >= start) & (frame_idx < end) & (labels >= 0)
    if m.sum() == 0:
        return -1
    votes = np.bincount(labels[m], weights=confs[m], minlength=NUM_CLASSES)
    winner = int(votes.argmax())
    if (labels[m] == winner).mean() < min_frac:
        return -1  # half the frames disagree with the "winner" -> skip it
    return winner


def build_dataset(poses_dir, labels_dir, window=45, stride=15):
    """Returns X (N, window, 132), y (N,)."""
    poses_dir, labels_dir = Path(poses_dir), Path(labels_dir)
    X, y = [], []
    for pose_file in sorted(poses_dir.glob("*.npz")):
        label_file = labels_dir / (pose_file.stem + "_labels.npz")
        if not label_file.exists():
            print(f"skip {pose_file.name}: no label file")
            continue
        pd = np.load(pose_file)
        ld = np.load(label_file)
        kp = pd["keypoints"]  # (T, 33, 4)
        if len(kp) < window:
            continue  # clip shorter than one window, not much we can do
        # frames where mediapipe found nobody are NaN — track them, then zero them
        valid = ~np.isnan(kp[:, 0, 0])
        kp = np.where(np.isnan(kp), 0.0, kp)
        feats = to_features(kp)

        for start in range(0, len(kp) - window + 1, stride):
            end = start + window
            if valid[start:end].mean() < 0.8:
                continue  # person missing for >20% of the window
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
