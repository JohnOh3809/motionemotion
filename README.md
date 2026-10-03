# MotionEmotion

MotionEmotion explores whether body movement can help estimate emotion when facial
expressions are limited or hard to read. The browser app tracks pose through a
webcam and displays emotion scores. The Python pipeline trains a model from
recorded clips.

The app starts with a heuristic classifier. It can also load a trained ONNX model.
The predictions are experimental and may not match how someone feels.

```text
motionemotion/
├── app/index.html       # webcam app
├── app/record.html      # training clip recorder
├── pipeline/            # labeling, pose extraction, training, and inference
└── train_on_colab.ipynb  # training in Colab
```

## Run the app

```bash
cd app
python -m http.server
```

Open http://localhost:8000 and select **Start camera**. Stand far enough back for
your upper body to fit in the frame.

MediaPipe Pose tracks 33 landmarks in the browser. The camera feed is processed
locally and is not uploaded. The app downloads its libraries and pose model on
first use.

Without a trained model, the classifier uses a rolling two-second window of
motion features, including speed, acceleration, arm spread, and head position.
**Calibrate neutral** records a three-second baseline for this classifier. The
app smooths the scores and shows **not sure** when the leading score is too low
or too close to the next one.

To use a trained model, place `motionemotion.onnx` and `motionemotion.onnx.json`
in `app/`. The app loads them automatically; the model badge shows which
classifier is active. Export both files with `infer.py --export-onnx`.

The app includes a manifest and service worker for installation in supported
browsers. Offline use depends on the required assets having been cached.

## Record training clips

Acted clips provide labels without running a facial-expression model. The label
is the emotion the participant intends to demonstrate.

1. Open `app/record.html`, choose an emotion, and record about five seconds of body movement.
2. Start with 10–20 clips per emotion. Vary the gestures and recording conditions; include different participants where possible.
3. Save the downloaded clips in one folder. The recorder names them with an emotion prefix, such as `happy_<timestamp>.webm`.

```bash
cd pipeline
pip install -r requirements.txt
python acted_label.py --videos /path/to/clips --out data_acted
python extract_poses.py --videos data_acted/videos --out data_acted/poses
python train.py --poses data_acted/poses --labels data_acted/labels --epochs 60
python infer.py --ckpt checkpoints/best.pt
python infer.py --ckpt checkpoints/best.pt --export-onnx motionemotion.onnx
```

Emotion subfolders such as `clips/happy/` also work. For training in Colab,
follow `train_on_colab.ipynb` to upload the pipeline and a ZIP of recorded clips.

## Train with facial-expression labels

The original training approach uses facial-expression predictions as labels for
body movement. Record videos with faces visible, generate labels with DeepFace,
and train the pose model to predict those labels. The trained pose model does
not use facial-expression predictions during inference.

```bash
cd pipeline
pip install -r requirements.txt
python extract_poses.py --videos data/videos --out data/poses
python auto_label.py --videos data/videos --out data/labels
python train.py --poses data/poses --labels data/labels
python infer.py --ckpt checkpoints/best.pt --export-onnx motionemotion.onnx
```

Pose coordinates are centered on the hips and scaled by torso length to reduce
variation from camera position and distance. Training uses windows of 45 frames
by default. Windows with insufficient label agreement or too many missing poses
are skipped, and the loss weights account for class imbalance.

Optional training settings:

- `--model transformer` selects the transformer instead of the default BiLSTM.
- `--merge5` maps disgusted to angry and surprised to fearful, producing five classes. The exported metadata tells the app which classes to display. Compare validation results before choosing this mapping.
- `--va-weight 0.3` adds an auxiliary valence/arousal loss during training. The auxiliary head is excluded from the exported classifier.

## BoLD dataset adapter

[BoLD](https://cydar.ist.psu.edu/emotionchallenge/dataset.php) contains movie clips
with body-language emotion annotations. `bold_adapter.py` maps its categories
to the project's seven classes and selects clips with a single annotated person.
Dataset access and citation details are available on the BoLD site.

After obtaining the dataset:

```bash
cd pipeline
python bold_adapter.py --bold /path/to/BOLD_public --out data_bold
python extract_poses.py --videos data_bold/videos --out data_bold/poses
python train.py --poses data_bold/poses --labels data_bold/labels --epochs 60
python infer.py --ckpt checkpoints/best.pt
```

The adapter's `--min-score` and `--min-margin` options filter ambiguous labels.
Stricter thresholds keep fewer clips. If the dataset is unavailable, the acted
clip workflow above can be used independently.

## Limitations and next steps

Facial-expression labels can be wrong, and an acted label describes an intended
performance rather than confirming someone's internal emotion. Some emotion
categories share similar movements. Check class counts and confusion matrices,
and evaluate on separate recordings and participants before relying on results.

Possible next steps include pretraining on a public dataset, fine-tuning on
personal recordings, and comparing the current models with a skeleton graph
network such as ST-GCN. An optional speech output would need the user's
confirmation before speaking a prediction.
