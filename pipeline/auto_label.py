"""Auto-label videos with emotions using a face expression model (DeepFace).

This implements the cross-modal supervision idea: when the face IS visible,
use it as a free labeler so the pose model can learn to predict the same
emotion from body movement alone.

Usage:
    python auto_label.py --videos data/videos --out data/labels [--every 5] [--min-conf 0.55]

For each video, saves <name>_labels.npz with:
    frame_idx: (N,) int32   — frames that were analyzed
    label:     (N,) int32   — index into EMOTIONS (-1 if no confident face)
    conf:      (N,) float32 — dominant emotion probability [0..1]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

# Same order everywhere in this project. DeepFace names map onto these.
EMOTIONS = ["happy", "sad", "angry", "fearful", "surprised", "disgusted", "neutral"]
DEEPFACE_MAP = {
    "happy": 0, "sad": 1, "angry": 2, "fear": 3,
    "surprise": 4, "disgust": 5, "neutral": 6,
}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def label_video(video_path: Path, out_dir: Path, every: int, min_conf: float) -> None:
    from deepface import DeepFace  # heavy import; keep local

    cap = cv2.VideoCapture(str(video_path))
    frame_idxs, labels, confs = [], [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every == 0:
            label, conf = -1, 0.0
            try:
                res = DeepFace.analyze(
                    frame, actions=["emotion"],
                    enforce_detection=True, silent=True,
                )
                emo = res[0]["emotion"]  # dict of percentages
                dom = max(emo, key=emo.get)
                conf = emo[dom] / 100.0
                if conf >= min_conf and dom in DEEPFACE_MAP:
                    label = DEEPFACE_MAP[dom]
            except Exception:
                pass  # no face found — leave unlabeled
            frame_idxs.append(idx)
            labels.append(label)
            confs.append(conf)
        idx += 1
    cap.release()

    out = out_dir / (video_path.stem + "_labels.npz")
    np.savez_compressed(
        out,
        frame_idx=np.array(frame_idxs, dtype=np.int32),
        label=np.array(labels, dtype=np.int32),
        conf=np.array(confs, dtype=np.float32),
    )
    n_ok = int((np.array(labels) >= 0).sum())
    print(f"{video_path.name}: {n_ok}/{len(labels)} confidently labeled -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=5, help="Analyze every Nth frame")
    ap.add_argument("--min-conf", type=float, default=0.55, help="Min dominant-emotion confidence")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = [p for p in sorted(Path(args.videos).iterdir()) if p.suffix.lower() in VIDEO_EXTS]
    if not videos:
        raise SystemExit(f"No videos found in {args.videos}")
    for v in videos:
        label_video(v, out_dir, args.every, args.min_conf)


if __name__ == "__main__":
    main()
