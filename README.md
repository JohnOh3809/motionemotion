# MotionEmotion

Express emotion through body movement — for people who can't (or don't) express it through facial expressions.

Two parts:

```
motionemotion/
├── app/index.html          # The app — installable PWA, works in any browser
├── app/record.html         # Record labeled training clips from your webcam
├── pipeline/               # Train your own model
└── train_on_colab.ipynb    # Train in the cloud (no big download on your PC)
```

**Recommended path:** record acted clips (section 2b) → train on Colab. It needs no
dataset download, no face model, and no big files on your computer. The BoLD route
(section 3) is left in for reference but its official download is currently broken.

## 1. The app

Serve the `app/` folder (`python -m http.server` inside it, then open http://localhost:8000)
or just open `app/index.html` directly. Click **Start camera**.

- **MediaPipe Pose** extracts 33 body keypoints in-browser (~30 fps, all local, nothing uploaded).
- A 2-second rolling window of keypoints is turned into interpretable motion features: movement energy, jerkiness, burst speed, arm openness, hands-above-shoulders, head drop, contraction, tremble, rising motion.
- A transparent heuristic classifier maps features → 7 emotions (happy, sad, angry, fearful, surprised, disgusted, neutral), smoothed over time.
- **Calibrate neutral** captures 3 seconds of your relaxed posture and re-centers the features — everyone's resting body is different, and this matters most for the app's target users.
- When probabilities are flat, the readout says **“not sure”** instead of guessing. For assistive use the app must never speak for you with false confidence.
- **Trained model auto-load:** if `motionemotion.onnx` (+ its `.json` sidecar, both produced by `infer.py --export-onnx`) sits next to `index.html` on a served page, the app switches from heuristics to the learned model automatically — the badge in the header shows which one is running.
- **Install as an app:** when served over http(s), Chrome/Edge shows an install icon in the address bar (⊕). Installed, it runs in its own window with an icon, and the shell works offline after first load.

Try it: bounce with open arms (happy), slump slowly with head down (sad), make sharp fast gestures (angry), pull in and tremble (fearful), sudden jump with arms up (surprised).

The heuristics are a starter brain. The pipeline below replaces them with a learned model.

## 2. Training pipeline (your idea: face model as the labeler)

Record videos of people moving **with their face visible**. A face-expression model labels each moment; the pose model learns to predict the same emotion from body movement alone. At inference time, no face needed.

```bash
cd pipeline
pip install -r requirements.txt

# 1. Extract 33 keypoints per frame from your videos
python extract_poses.py --videos data/videos --out data/poses

# 2. Auto-label using DeepFace's emotion model (7 classes, matches the app)
python auto_label.py --videos data/videos --out data/labels

# 3. Train a BiLSTM (or --model transformer) on 45-frame keypoint windows
python train.py --poses data/poses --labels data/labels

# 4. Run live, or export ONNX (+ metadata sidecar) to plug into the web app
python infer.py --ckpt checkpoints/best.pt
python infer.py --ckpt checkpoints/best.pt --export-onnx motionemotion.onnx
```

Two optional training upgrades:

- `--merge5` trains 5 classes instead of 7, merging the two that are genuinely hard to read from body movement (disgusted→angry, surprised→fearful). Recommended when accuracy matters more than coverage. The app adapts automatically via the ONNX metadata sidecar.
- `--va-weight 0.3` adds an auxiliary **valence/arousal** regression head during training. Body movement encodes arousal far more reliably than discrete categories, so this extra signal regularizes the encoder. Training-only — the exported model is unchanged in shape.

Keypoints are normalized (centered on hips, scaled by torso length) so the model is camera-position invariant. Windows without a confident, consistent face label are dropped; class weights handle the neutral-heavy imbalance.

## 2b. Record your own acted clips (recommended)

The fastest way to a real trained model — and it works with limited disk since clips are small.

1. Open `app/record.html` in Chrome. Pick an emotion, hit record, and **act it with your body** for ~5s (exaggerate). Each clip auto-downloads named `happy_<timestamp>.webm`, etc.
2. Aim for ~10–20 varied clips per emotion. Recruit friends for variety — different bodies help the model generalize.
3. Put every clip in one folder, then label + train (the filename prefix is the label — no face model needed):

```bash
cd pipeline
python acted_label.py --videos /path/to/your/clips --out data_acted
python extract_poses.py --videos data_acted/videos --out data_acted/poses
python train.py --poses data_acted/poses --labels data_acted/labels --epochs 60
python infer.py --ckpt checkpoints/best.pt        # test live
```

Clips can also be organized as subfolders (`clips/happy/…`, `clips/sad/…`) instead of prefixes. Or train it all in the cloud with `train_on_colab.ipynb` so nothing heavy touches your machine.

## 3. Training on the BoLD public dataset (reference — download currently broken)

> ⚠️ As of this writing the official BoLD download link 404s on Penn State's server. Kept here in case they restore it. Prefer section 2b.

BoLD has ~9.8k movie clips with crowdsourced body-emotion annotations. Register (free) and download `BOLD_public` from https://cydar.ist.psu.edu/emotionchallenge/dataset.php (citation of the ARBEE paper required if you publish).

```bash
cd pipeline
# map BoLD's 26 emotion categories -> our 7, keep clear single-person clips
python bold_adapter.py --bold /path/to/BOLD_public --out data_bold

# then the normal pipeline
python extract_poses.py --videos data_bold/videos --out data_bold/poses
python train.py --poses data_bold/poses --labels data_bold/labels --epochs 60
python infer.py --ckpt checkpoints/best.pt
```

Tune `--min-score` / `--min-margin` in `bold_adapter.py` to trade dataset size vs. label clarity. Expect a class-imbalanced, hard dataset — 40-50% val accuracy over 7 classes is already far above chance (14%) for body-only signals. Best recipe: pretrain on BoLD, then fine-tune on your own face-labeled or acted recordings.

## Ideas to make it better

- **Pretrain on a public dataset first.** The BoLD (Body Language Dataset) and GEMEP corpora have human-annotated body-emotion labels — pretrain there, fine-tune with your face-labeled data.
- **Acted data bootstrapping.** Ask volunteers to *act* each emotion for 30s clips — instant clean labels, great for a first model when face-labeling yields mostly neutral.
- **Speak for me.** Add text-to-speech so the app can say the emotion out loud — with an explicit confirm step, never automatically.
- **Deeper personal calibration.** The app now re-centers on your neutral; the next level is a per-user "show me your happy / sad / angry" session that fine-tunes the learned model itself.
- **Graph model.** Upgrade LSTM → ST-GCN (spatial-temporal graph conv), the standard for skeleton-based action recognition.

Done and in the code: ~~valence/arousal auxiliary training~~ (`--va-weight`), ~~class merging~~ (`--merge5`), ~~neutral calibration~~, ~~uncertainty display~~ (the "not sure" state).

## Honest caveats

- Face labels are noisy weak supervision — filter aggressively (already done via confidence + majority voting) and expect to need hours of video.
- "Disgusted" and "surprised" are genuinely hard to read from body movement; expect confusion between them. That's what `--merge5` is for, if accuracy matters more than coverage.
- For assistive use, always show probabilities rather than a single confident answer — the app does this, and says "not sure" when the signal is ambiguous. Keep it that way.
