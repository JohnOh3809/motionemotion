"""Extract body keypoints from videos using MediaPipe Pose.

Usage:
    python extract_poses.py --videos data/videos --out data/poses

For each video, saves <name>.npz with:
    keypoints: (T, 33, 4) float32 — x, y, z, visibility (normalized coords)
    timestamps: (T,) float32 — seconds
    fps: float
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def extract(video_path: Path, out_dir: Path) -> None:
    pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    keypoints, timestamps = [], []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.pose_landmarks:
            pts = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark],
                dtype=np.float32,
            )
        else:
            pts = np.full((33, 4), np.nan, dtype=np.float32)  # no person detected
        keypoints.append(pts)
        timestamps.append(frame_idx / fps)
        frame_idx += 1

    cap.release()
    pose.close()

    out = out_dir / (video_path.stem + ".npz")
    np.savez_compressed(
        out,
        keypoints=np.stack(keypoints) if keypoints else np.zeros((0, 33, 4), np.float32),
        timestamps=np.array(timestamps, dtype=np.float32),
        fps=fps,
    )
    print(f"{video_path.name}: {frame_idx} frames -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="Directory of input videos")
    ap.add_argument("--out", required=True, help="Output directory for .npz pose files")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = [p for p in sorted(Path(args.videos).iterdir()) if p.suffix.lower() in VIDEO_EXTS]
    if not videos:
        raise SystemExit(f"No videos found in {args.videos}")
    for v in videos:
        extract(v, out_dir)


if __name__ == "__main__":
    main()
