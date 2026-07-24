# MotionEmotion

Some people can't (or don't) show emotion on their face — but everyone moves.
This project tries to read emotion from body movement alone: an app that watches
your pose through the webcam and tells you what your body is saying, plus a
pipeline to train it on your own recordings.

```
motionemotion/
├── app/index.html          # the app — installable PWA, works in any browser
├── app/record.html         # record labeled training clips from your webcam
├── pipeline/               # train your own model
└── train_on_colab.ipynb    # train in the cloud (nothing big touches your PC)
```

**Fastest route to a real model:** record acted clips (section 2b) → train on Colab.
No dataset downloads, no face model, no giant files. The BoLD route (section 3) is
still written up but their download link is currently dead, so.

## 1. The app

Serve the `app/` folder (`python -m http.server` inside it, then http://localhost:8000)
— or just double-click `index.html` for a quick look. Hit **Start camera**.

What's going on under the hood:

- MediaPipe Pose pulls 33 body keypoints per frame, right in the browser (~30 fps, all local, nothing gets uploaded anywhere).
- A rolling 2-second window becomes motion features you could explain to a friend: energy, jerkiness, burst speed, arm openness, hands-above-shoulders, head drop, contraction, tremble, rise.
- A hand-tuned classifier maps those to 7 emotions and smooths over time. It's a starter brain — the pipeline exists to replace it.
- **Calibrate neutral**: 3 seconds of you standing still teaches it what *your* resting body looks like, so it reacts to change from your baseline instead of judging your posture. Everyone fidgets differently; for the people this app is meant for, that matters a lot.
- When the numbers are mushy it says **"not sure"** instead of bluffing. An assistive app that guesses confidently and wrong is speaking over the person using it. Non-negotiable.
- **Trained model auto-load**: put `motionemotion.onnx` (+ its `.json`, both from `infer.py --export-onnx`) next to `index.html` on a served page and the app switches from heuristics to your model on its own. The badge up top shows which brain is running.
- **Install it**: served over http(s), Chrome/Edge show an install icon (⊕) in the address bar. Then it lives in your dock with its own window and icon, and works offline after the first load.

Party trick suggestions: bounce with open arms (happy), slump slowly with your head down (sad), sharp fast gestures (angry), pull in and tremble (fearful), sudden jump with arms up (surprised).

## 2. Training pipeline (the original idea: face as the teacher)

Record people moving **with their face visible**. A face-expression model labels each
moment, and the pose model learns to predict the same emotion from body movement
alone. At inference time no face is needed — the face was just the teacher.

```bash
cd pipeline
pip install -r requirements.txt

# 1. videos -> 33 keypoints per frame
python extract_poses.py --videos data/videos --out data/poses

# 2. let DeepFace label the frames it's confident about
python auto_label.py --videos data/videos --out data/labels

# 3. train a BiLSTM (or --model transformer) on 45-frame windows
python train.py --poses data/poses --labels data/labels

# 4. run it live, or export for the web app
python infer.py --ckpt checkpoints/best.pt
python infer.py --ckpt checkpoints/best.pt --export-onnx motionemotion.onnx
```

Two training flags worth knowing:

- `--merge5` — "disgusted" and "surprised" are basically unreadable from body movement alone, so this folds them into their nearest neighbors (disgusted→angry, surprised→fearful) and trains 5 classes instead of 7. The app adapts by itself via the metadata sidecar. Recommended once you care about accuracy.
- `--va-weight 0.3` — adds a small valence/arousal regression head during training. Bodies broadcast *arousal* much more reliably than discrete categories, and giving the encoder that side-task helps it. Training-only; the exported model is unchanged.

Keypoints are hip-centered and torso-scaled, so camera distance and position don't
matter. Windows without a clear, confident label get dropped, and class weights deal
with the fact that face-labeled data comes out overwhelmingly neutral (people mostly
just stand there).

## 2b. Record your own acted clips (do this one)

The label comes free because you know what you acted. Small files, quick loop, works
with limited disk.

1. Open `app/record.html`. Pick an emotion, hit record, act it with your whole body for ~5s. Exaggerate — nobody's watching. Each clip downloads itself as `happy_<timestamp>.webm` etc., label baked into the filename.
2. Get ~10–20 varied clips per emotion. Rope in friends — different bodies help it generalize.
3. Everything into one folder, then:

```bash
cd pipeline
python acted_label.py --videos /path/to/your/clips --out data_acted
python extract_poses.py --videos data_acted/videos --out data_acted/poses
python train.py --poses data_acted/poses --labels data_acted/labels --epochs 60
python infer.py --ckpt checkpoints/best.pt        # see it work
```

Subfolders (`clips/happy/…`) work too if you prefer that over filename prefixes. Or
skip local entirely and use `train_on_colab.ipynb` — zip your clips, upload, train on
a free GPU, download a few-MB model file.

## 3. BoLD public dataset (reference — download currently 404s)

> Penn State's official download link is broken as of this writing. Keeping the
> instructions in case they fix it. Use section 2b instead.

BoLD is ~9.8k movie clips with crowdsourced body-language emotion annotations.
Register (free) at https://cydar.ist.psu.edu/emotionchallenge/dataset.php — cite the
ARBEE paper if you publish anything.

```bash
cd pipeline
# their 26 emotion categories -> our 7, keep clear single-person clips
python bold_adapter.py --bold /path/to/BOLD_public --out data_bold

python extract_poses.py --videos data_bold/videos --out data_bold/poses
python train.py --poses data_bold/poses --labels data_bold/labels --epochs 60
python infer.py --ckpt checkpoints/best.pt
```

`--min-score` / `--min-margin` in the adapter trade dataset size against label
clarity. Expect it to be hard: 40–50% val accuracy over 7 classes sounds bad until
you remember chance is 14% and you're only looking at bodies. The dream recipe is
pretrain on BoLD, fine-tune on your own clips.

## Ideas parked for later

- **Pretrain on a public corpus** (BoLD, GEMEP) and fine-tune on personal recordings.
- **"Speak for me"** — text-to-speech saying the emotion out loud, with an explicit confirm step. Never automatic.
- **Deeper personal fine-tuning** — a "show me your happy / sad / angry" session that adapts the learned model itself, not just the feature baseline.
- **ST-GCN** — the proper architecture for skeleton sequences, once there's enough data to feed it.

Already done and in the code: valence/arousal aux training (`--va-weight`), class
merging (`--merge5`), neutral calibration, and the "not sure" state.

## Honest caveats

- Face labels are noisy, weak supervision. Filter hard (the code does) and expect to want hours of video.
- Disgust and surprise from body movement alone are close to hopeless — that's what `--merge5` is for.
- For assistive use, show probabilities and admit uncertainty. The app does. Keep it that way.
