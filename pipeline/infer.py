"""Run the trained model live on webcam (or a video file), and export to ONNX.

Usage:
    python infer.py --ckpt checkpoints/best.pt                # webcam
    python infer.py --ckpt checkpoints/best.pt --video x.mp4  # video file
    python infer.py --ckpt checkpoints/best.pt --export-onnx model.onnx
"""
import argparse
from collections import deque

import cv2
import numpy as np
import torch

from dataset import EMOTIONS, to_features
from model import build_model


def load(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["window"]


def export_onnx(model, window, out_path):
    from dataset import FEAT_DIM
    dummy = torch.zeros(1, window, FEAT_DIM)
    torch.onnx.export(model.cpu(), dummy, out_path,
                      input_names=["keypoints"], output_names=["logits"],
                      dynamic_axes={"keypoints": {0: "batch"}})
    print(f"exported ONNX -> {out_path} (usable in the web app via onnxruntime-web)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--video", default=None, help="Video file; default = webcam 0")
    ap.add_argument("--export-onnx", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, window = load(args.ckpt, device)

    if args.export_onnx:
        export_onnx(model, window, args.export_onnx)
        return

    import mediapipe as mp
    pose = mp.solutions.pose.Pose(model_complexity=1)
    cap = cv2.VideoCapture(args.video if args.video else 0)
    buf = deque(maxlen=window)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        text = "no person"
        if res.pose_landmarks:
            pts = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                            for lm in res.pose_landmarks.landmark], dtype=np.float32)
            buf.append(pts)
            if len(buf) == window:
                feats = to_features(np.stack(buf))          # (window, FEAT_DIM)
                x = torch.from_numpy(feats[None]).to(device)
                with torch.no_grad():
                    probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
                top = int(probs.argmax())
                text = f"{EMOTIONS[top]} {probs[top]*100:.0f}%"
        cv2.putText(frame, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 120), 2)
        cv2.imshow("MotionEmotion", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
