# FSLAMES video/image-to-TFLite pipeline

This package supports five participants, static or dynamic signs, and one- or
two-hand inputs. It keeps participants—not frames—from leaking between
training, validation, and testing. Augmentation is applied only to the training
participants.

## Setup

Use Python 3.10 or 3.11:

```powershell
py -3.11 -m venv .venv-ml
.\.venv-ml\Scripts\python.exe -m pip install -r requirements.txt
```

Download `hand_landmarker.task` from the official MediaPipe model collection.

## Raw dataset folders

Keep each model type in a separate root. Under each sign label, place one
folder for each of the five participants:

```text
raw_static_single/
  A/P01/A_right.jpg
  A/P01/A_left.jpg
  A/P02/A_right.jpg
  A/P02/A_left.jpg
  ...
  A/P05/A_right.jpg
  A/P05/A_left.jpg
  B/P01/B_right.mp4
  B/P01/B_left.mp4
  ...

raw_static_both/
  FAMILY/P01/FAMILY_both.jpg
  ...

raw_dynamic_single/
  HELLO/P01/HELLO_right.mp4
  HELLO/P01/HELLO_left.mp4
  ...

raw_dynamic_both/
  THANK_YOU/P01/THANK_YOU_both.mp4
  ...
```

The default layout is `label/participant/file`. Use
`--layout participant/label` if your first two folders are reversed.
Filenames must contain exactly one role token: `_right`, `_left`, or `_both`.

## Extract CSV files

Static one-hand images/videos:

```powershell
.\.venv-ml\Scripts\python.exe extract_landmarks.py raw_static_single `
  --model hand_landmarker.task --output static_single.csv `
  --num-hands 1 --sign-type static
```

Static two-hand images/videos:

```powershell
.\.venv-ml\Scripts\python.exe extract_landmarks.py raw_static_both `
  --model hand_landmarker.task --output static_both.csv `
  --num-hands 2 --sign-type static
```

Dynamic one-hand videos:

```powershell
.\.venv-ml\Scripts\python.exe extract_landmarks.py raw_dynamic_single `
  --model hand_landmarker.task --output dynamic_single.csv `
  --num-hands 1 --sign-type dynamic --frame-skip 2
```

Dynamic two-hand videos:

```powershell
.\.venv-ml\Scripts\python.exe extract_landmarks.py raw_dynamic_both `
  --model hand_landmarker.task --output dynamic_both.csv `
  --num-hands 2 --sign-type dynamic --frame-skip 2
```

One-hand CSVs contain 63 values per frame. Two-hand CSVs contain 126 values.
For two-hand samples, hands are consistently ordered and both are translated
relative to the first hand's wrist, preserving their spatial relationship.

## Validate coverage and capture consistency

Create `cvi_manifest.json` from `cvi_manifest.example.json`. Release training
requires two independent SPED/FSL experts to approve every label. Then run:

```powershell
.\.venv-ml\Scripts\python.exe validate_dataset.py static_single.csv
```

The same validation runs automatically during training. It requires at least
five participants. For every one-hand sign, every participant must have exactly
one right source and one left source. For every two-hand sign, every participant
must have exactly one `both` source. It also checks detection confidence,
ordinary lighting, centered hands, mid-distance palm scale, and at least eight
detected frames per dynamic video. At least 80% of sampled frames in each source
must pass the capture-consistency checks.

## Inspect and train static models

```powershell
.\.venv-ml\Scripts\python.exe train_landmark_classifier.py `
  static_single.csv --inspect-only

.\.venv-ml\Scripts\python.exe train_landmark_classifier.py `
  static_single.csv --output-dir output_static_single `
  --augmentation-copies 4 --cvi-manifest cvi_manifest.json
```

Use the same trainer with `static_both.csv` and a different output directory.
The trainer automatically detects 63 versus 126 features.

## Inspect and train dynamic models

```powershell
.\.venv-ml\Scripts\python.exe train_sequence_classifier.py `
  dynamic_single.csv --inspect-only

.\.venv-ml\Scripts\python.exe train_sequence_classifier.py `
  dynamic_single.csv --output-dir output_dynamic_single `
  --sequence-length 24 --augmentation-copies 12 `
  --cvi-manifest cvi_manifest.json
```

Use the same sequence trainer for `dynamic_both.csv`. It resamples every video
to a fixed temporal length, adds palm-normalized center displacement and
velocity, and applies augmentation consistently across the full sequence.
The exported target-aware policy is ten percentage points more lenient for a
correct sign while still rejecting a different top-1 sign by margin.

For non-release experiments only, `--allow-unvalidated-for-development` marks
the output as unvalidated. Such a bundle cannot be installed into the app.

## Optional Roboflow audit

```powershell
python -m pip install -r requirements-roboflow.txt
$env:ROBOFLOW_API_KEY = "your-key"
python roboflow_audit.py raw_dynamic_single --model-id fsl-fnlzs/3
```

This produces a review report only. It never relabels data automatically and
Roboflow is never called by the offline mobile runtime.

## Five-person evaluation

With five participants, the default split uses three people for training, one
for validation, and one for testing. No frames from a held-out person enter the
training data. The generated accuracy, precision, recall, F1-score, and
confusion matrix therefore measure performance on an unseen participant.

This remains a small prototype dataset. Augmentation improves robustness but
does not create new people or guarantee real-world accuracy.

## Flutter installation and fallback

The bundle-v2 installer accepts static/dynamic and one/two-hand model slots:

```powershell
.\.venv-ml\Scripts\python.exe install_mobile_bundle.py `
  output_static_single\mobile_bundle
```

Run it once for each trained slot. The Android runtime validates model kind,
preprocessing, input tensor, labels, bundle version, and expert-CVI status. A
missing model safely falls back to the offline reference/deviation scorer.

## Outputs

Each trainer produces a Keras model, verified TFLite model, model metadata,
training history, confusion matrix, and `mobile_bundle/`. TFLite export passes
only when its verification prediction matches the Keras model prediction.
