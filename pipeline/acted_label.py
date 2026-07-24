"""Label acted clips from their filenames. No face model, no waiting.

If you KNOW the emotion because you acted it, the label is free. Two layouts
both work:

  A) subfolders:       clips/happy/take1.mp4, clips/sad/take2.mp4, ...
  B) filename prefix:  clips/happy_01.webm, clips/sad_02.webm, ...

record.html spits out option (B) automatically, so that's the usual path.

This flattens everything into <out>/videos/ and writes <out>/labels/ in the
same format auto_label.py uses, labeling every frame with the acted emotion.

    python acted_label.py --videos clips --out data_acted
    python extract_poses.py --videos data_acted/videos --out data_acted/poses
    python train.py --poses data_acted/poses --labels data_acted/labels
"""
import argparse
import os
import shutil
from pathlib import Path

import numpy as np

EMOTIONS = ["happy", "sad", "angry", "fearful", "surprised", "disgusted", "neutral"]
EMO_INDEX = {e: i for i, e in enumerate(EMOTIONS)}
# people name folders all kinds of ways, accept the obvious ones
ALIASES = {"fear": "fearful", "scared": "fearful", "surprise": "surprised",
           "disgust": "disgusted", "mad": "angry", "calm": "neutral"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def emotion_of(path: Path, root: Path):
    # subfolder name wins; otherwise whatever comes before the first _ or -
    rel = path.relative_to(root)
    if len(rel.parts) > 1:
        cand = rel.parts[0].lower()
    else:
        cand = path.stem.lower().replace("-", "_").split("_")[0]
    cand = ALIASES.get(cand, cand)
    return EMO_INDEX.get(cand)


def frame_count(path: Path) -> int:
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n > 0:
            return n
    except Exception:
        pass
    return 300  # ~10s guess if cv2 can't tell us; close enough for labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy", action="store_true", help="Copy instead of symlink")
    args = ap.parse_args()

    root = Path(args.videos)
    out_videos = Path(args.out) / "videos"
    out_labels = Path(args.out) / "labels"
    out_videos.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    clips = [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    if not clips:
        raise SystemExit(f"No videos found under {root}")

    kept, skipped, counts = 0, [], np.zeros(len(EMOTIONS), dtype=int)
    for clip in sorted(clips):
        emo = emotion_of(clip, root)
        if emo is None:
            skipped.append(clip.name)
            continue
        name = f"{EMOTIONS[emo]}_{clip.stem}".replace(" ", "")
        dst = out_videos / (name + clip.suffix.lower())
        if not dst.exists():
            (shutil.copy2 if args.copy else lambda s, d: os.symlink(Path(s).resolve(), d))(clip, dst)

        # every 5th frame gets a label row, conf 1.0 — we acted it, we're sure
        n = frame_count(clip)
        idx = np.arange(0, n, 5, dtype=np.int32)
        np.savez_compressed(
            out_labels / (name + "_labels.npz"),
            frame_idx=idx,
            label=np.full(len(idx), emo, dtype=np.int32),
            conf=np.ones(len(idx), dtype=np.float32),
        )
        kept += 1
        counts[emo] += 1

    print(f"labeled {kept} clips -> {args.out}")
    print("class counts: " + ", ".join(f"{e}={c}" for e, c in zip(EMOTIONS, counts)))
    if skipped:
        print(f"\nskipped {len(skipped)} clip(s) with no recognizable emotion name:")
        print("  " + ", ".join(skipped[:10]) + (" ..." if len(skipped) > 10 else ""))
        print("  -> put them in a subfolder named after the emotion, or prefix the filename")
    zero = [EMOTIONS[i] for i, c in enumerate(counts) if c == 0]
    if zero:
        print(f"\nno clips yet for: {', '.join(zero)} — record some for balance.")


if __name__ == "__main__":
    main()
