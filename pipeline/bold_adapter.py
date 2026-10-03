"""Adapt the BoLD (Body Language Dataset) to the MotionEmotion pipeline.

BoLD: https://cydar.ist.psu.edu/emotionchallenge/dataset.php  (free registration
required). After downloading, you have BOLD_public/ with annotations/ and videos/.

BoLD annotates 26 emotion categories per person-instance. This script:
  1. maps the 26 categories onto our 7 emotions (ambiguous categories are ignored),
  2. keeps only clips with a single annotated person (our pose extractor tracks one),
  3. keeps only instances whose winning emotion is clear (score + margin thresholds),
  4. symlinks each kept clip into <out>/videos/ under a flat, unique name,
  5. writes <out>/labels/<name>_labels.npz in the same format auto_label.py produces.

Then the normal pipeline takes over:
    python bold_adapter.py --bold /path/to/BOLD_public --out data_bold
    python extract_poses.py --videos data_bold/videos --out data_bold/poses
    python train.py --poses data_bold/poses --labels data_bold/labels

Usage:
    python bold_adapter.py --bold BOLD_public --out data_bold \
        [--splits train val] [--min-score 0.25] [--min-margin 0.08] [--copy]
"""
import argparse
import csv
import os
import shutil
from pathlib import Path

import numpy as np

EMOTIONS = ["happy", "sad", "angry", "fearful", "surprised", "disgusted", "neutral"]

BOLD_CATEGORIES = [
    "Peace", "Affection", "Esteem", "Anticipation", "Engagement", "Confidence",
    "Happiness", "Pleasure", "Excitement", "Surprise", "Sympathy", "Doubt/Confusion",
    "Disconnect", "Fatigue", "Embarrassment", "Yearning", "Disapproval", "Aversion",
    "Annoyance", "Anger", "Sensitivity", "Sadness", "Disquietment", "Fear",
    "Pain", "Suffering",
]

# 26 BoLD categories -> our 7 (None = too ambiguous for body movement; ignored)
CATEGORY_MAP = {
    "Happiness": "happy", "Pleasure": "happy", "Excitement": "happy", "Affection": "happy",
    "Sadness": "sad", "Suffering": "sad", "Pain": "sad", "Fatigue": "sad",
    "Anger": "angry", "Annoyance": "angry",
    "Fear": "fearful", "Disquietment": "fearful",
    "Surprise": "surprised",
    "Aversion": "disgusted", "Disapproval": "disgusted",
    "Peace": "neutral", "Disconnect": "neutral",
    # ignored: Esteem, Anticipation, Engagement, Confidence, Sympathy,
    # Doubt/Confusion, Embarrassment, Yearning, Sensitivity
}

HEADER = (["video", "person_id", "min_frame", "max_frame"] + BOLD_CATEGORIES
          + ["valence", "arousal", "dominance", "gender", "age", "ethnicity",
             "annotation_confidence"])


def flat_name(video_rel: str) -> str:
    """'003/IzvOYVMltkI.mp4/0114.mp4' -> '003_IzvOYVMltkI_0114'"""
    return video_rel.replace(".mp4", "").replace("/", "_").replace(" ", "")


def video_frame_count(path: Path) -> int:
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n > 0:
            return n
    except Exception:
        pass
    return 300  # fallback when video metadata is unavailable


def load_split(bold: Path, split: str):
    ann = bold / "annotations" / f"{split}.csv"
    if not ann.exists():
        print(f"skip split '{split}': {ann} not found")
        return []
    rows = []
    with open(ann, newline="") as f:
        for r in csv.reader(f):
            if len(r) < len(HEADER) - 1:
                continue
            rows.append(dict(zip(HEADER, r)))
    return rows


def score_row(row):
    """Return (label_idx, score, margin) after mapping 26 -> 7."""
    agg = {e: 0.0 for e in EMOTIONS}
    for cat, emo in CATEGORY_MAP.items():
        agg[emo] += float(row[cat])
    ranked = sorted(agg.items(), key=lambda kv: -kv[1])
    (top, s1), (_, s2) = ranked[0], ranked[1]
    return EMOTIONS.index(top), s1, s1 - s2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bold", required=True, help="Path to BOLD_public directory")
    ap.add_argument("--out", required=True, help="Output dir (creates videos/ and labels/)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--min-score", type=float, default=0.25,
                    help="Min aggregated score of the winning emotion")
    ap.add_argument("--min-margin", type=float, default=0.08,
                    help="Winner must beat runner-up by this much")
    ap.add_argument("--copy", action="store_true",
                    help="Copy videos instead of symlinking")
    args = ap.parse_args()

    bold = Path(args.bold)
    out_videos = Path(args.out) / "videos"
    out_labels = Path(args.out) / "labels"
    out_videos.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in args.splits:
        rows += load_split(bold, split)
    print(f"loaded {len(rows)} annotated instances")

    # keep only single-person clips (our extractor tracks one person)
    by_video = {}
    for r in rows:
        by_video.setdefault(r["video"], []).append(r)
    singles = [v[0] for v in by_video.values() if len(v) == 1]
    print(f"{len(singles)} single-person clips")

    kept, counts = 0, np.zeros(len(EMOTIONS), dtype=int)
    for row in singles:
        label, score, margin = score_row(row)
        if score < args.min_score or margin < args.min_margin:
            continue
        src = bold / "videos" / row["video"]
        if not src.exists():
            continue

        name = flat_name(row["video"])
        dst = out_videos / (name + ".mp4")
        if not dst.exists():
            if args.copy:
                shutil.copy2(src, dst)
            else:
                os.symlink(src.resolve(), dst)

        # label every 5th frame of the whole clip with the clip-level emotion
        n = video_frame_count(src)
        idx = np.arange(0, n, 5, dtype=np.int32)
        conf = float(np.clip(score, 0.0, 1.0))
        np.savez_compressed(
            out_labels / (name + "_labels.npz"),
            frame_idx=idx,
            label=np.full(len(idx), label, dtype=np.int32),
            conf=np.full(len(idx), conf, dtype=np.float32),
        )
        kept += 1
        counts[label] += 1

    print(f"\nkept {kept} clips -> {args.out}")
    print("class counts: " + ", ".join(f"{e}={c}" for e, c in zip(EMOTIONS, counts)))
    print("\nnext steps:")
    print(f"  python extract_poses.py --videos {args.out}/videos --out {args.out}/poses")
    print(f"  python train.py --poses {args.out}/poses --labels {args.out}/labels")


if __name__ == "__main__":
    main()
