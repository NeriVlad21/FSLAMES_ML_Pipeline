# FSLAMES video/image-to-TFLite pipeline

This package supports five participants, static or dynamic signs, and one- or
two-hand inputs. It keeps participants—not frames—from leaking between
training, validation, and testing. Augmentation is applied only to the training
participants.

## Setup

Use Python 3.10 or 3.11:

```powershell
py -3.11 -m venv .venv-ml
.\.venv-ml\Scripts\python.exe -m pip install -r tools\ml\requirements.txt
```

The CLI scripts import the `fslames_ml` package. Running them from their own
folder works on a fresh checkout; to import the package from anywhere (tests,
notebooks, other folders), install it in editable mode:

```powershell
.\.venv-ml\Scripts\python.exe -m pip install -e "tools\ml[dev]"
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
.\.venv-ml\Scripts\python.exe tools\ml\extract_landmarks.py raw_static_single `
  --model hand_landmarker.task --output static_single.csv `
  --num-hands 1 --sign-type static
```

Static two-hand images/videos:

```powershell
.\.venv-ml\Scripts\python.exe tools\ml\extract_landmarks.py raw_static_both `
  --model hand_landmarker.task --output static_both.csv `
  --num-hands 2 --sign-type static
```

Dynamic one-hand videos:

```powershell
.\.venv-ml\Scripts\python.exe tools\ml\extract_landmarks.py raw_dynamic_single `
  --model hand_landmarker.task --output dynamic_single.csv `
  --num-hands 1 --sign-type dynamic --frame-skip 2
```

Dynamic two-hand videos:

```powershell
.\.venv-ml\Scripts\python.exe tools\ml\extract_landmarks.py raw_dynamic_both `
  --model hand_landmarker.task --output dynamic_both.csv `
  --num-hands 2 --sign-type dynamic --frame-skip 2
```

One-hand CSVs contain 63 values per frame. Two-hand CSVs contain 126 values.
For two-hand samples, hands are consistently ordered (MediaPipe `Left` then
`Right`; if both hands receive the same handedness label, the hand with the
smaller wrist x comes first) and both are translated relative to the first
hand's wrist, preserving their spatial relationship. Static two-hand signs are
still single-frame samples: they use the static trainer and never need video
or temporal behaviour.

## Validate coverage and capture consistency

Run this before either trainer:

```powershell
.\.venv-ml\Scripts\python.exe tools\ml\validate_dataset.py static_single.csv
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
.\.venv-ml\Scripts\python.exe tools\ml\train_landmark_classifier.py `
  static_single.csv --inspect-only

.\.venv-ml\Scripts\python.exe tools\ml\train_landmark_classifier.py `
  static_single.csv --output-dir output_static_single `
  --augmentation-copies 4
```

Use the same trainer with `static_both.csv` and a different output directory.
The trainer automatically detects 63 versus 126 features.

## Inspect and train dynamic models

```powershell
.\.venv-ml\Scripts\python.exe tools\ml\train_sequence_classifier.py `
  dynamic_single.csv --inspect-only

.\.venv-ml\Scripts\python.exe tools\ml\train_sequence_classifier.py `
  dynamic_single.csv --output-dir output_dynamic_single `
  --sequence-length 24 --augmentation-copies 12
```

Use the same sequence trainer for `dynamic_both.csv`. It resamples every video
to a fixed temporal length and applies each spatial augmentation consistently
across the full sequence and both hands.

Because wrist-relative landmarks alone lose where the hand travels, each
dynamic frame also carries four movement values after the landmarks:
`hand_center_dx, hand_center_dy` (hand-center displacement from the first
frame divided by the sequence's median palm size) and `hand_center_vx,
hand_center_vy` (their per-step velocity). The model input is therefore
`[1, T, 67]` for one hand and `[1, T, 130]` for two hands. The exact order and
formulas are written to `input_contract` in the manifest, and export fails if
the TFLite input shape differs from that contract.

## Expert validation (CVI) gate

Training a release model requires `--cvi-manifest` naming two independent
experts, with both approving every trained label (see
`cvi_manifest.example.json`). For unvalidated experiments only, pass
`--allow-unvalidated-for-development`; such bundles are marked
`release_ready: false` and the installer refuses them unless
`--allow-development-bundle` is given. `--inspect-only` reports the CVI status
without enforcing it.

## Target-aware recognition policy

An attempt at a requested sign is accepted only when the requested sign is the
model's top-1 prediction, its probability is at least 30%
(`--target-threshold 0.30`), and it leads the nearest other known sign by at
least `--min-margin` (default 0.06). A recognizable wrong sign is therefore
rejected even if the requested sign still scores above 30%. The policy is
written to the manifest as `decision_policy`, together with its measured
correct-sign acceptance and wrong-sign false-acceptance rates on the
validation participants; test-participant rates are in `model_metadata.json`.

## Five-person evaluation

With five participants, the default split uses three people for training, one
for validation, and one for testing. No frames from a held-out person enter the
training data. The generated accuracy, precision, recall, F1-score, and
confusion matrix therefore measure performance on an unseen participant.

This remains a small prototype dataset. Augmentation improves robustness but
does not create new people or guarantee real-world accuracy.

## Flutter installation and fallback

There are four independent model slots, each with its own file names so all
four can be installed side by side:

| Slot | Trainer | Input shape | Files |
| --- | --- | --- | --- |
| `static_single` | landmark | `[1, 63]` | `fslames_static_single_classifier.tflite`, `fslames_static_single_manifest.json` |
| `static_both` | landmark | `[1, 126]` | `fslames_static_both_classifier.tflite`, `fslames_static_both_manifest.json` |
| `dynamic_single` | sequence | `[1, T, 67]` | `fslames_dynamic_single_classifier.tflite`, `fslames_dynamic_single_manifest.json` |
| `dynamic_both` | sequence | `[1, T, 130]` | `fslames_dynamic_both_classifier.tflite`, `fslames_dynamic_both_manifest.json` |

```powershell
.\.venv-ml\Scripts\python.exe tools\ml\install_mobile_bundle.py `
  output_static_single\mobile_bundle
```

The installer validates bundle version 2, the slot, model kind, input
contract, labels, decision policy, and CVI release status. The Flutter runtime
must read these per-slot files and bundle version 2. If any model is missing,
rejected, or fails to load, FSLAMES continues using its existing CVI-validated
reference/deviation scorer.

## Optional Roboflow dataset audit

Roboflow is never required for training or by the mobile app; normal FSLAMES
practice stays offline. As a second opinion on raw media labels, you can
compare folder labels against the Roboflow model `fsl-fnlzs/3`:

```powershell
.\.venv-ml\Scripts\python.exe -m pip install -r tools\ml\requirements-roboflow.txt
$env:ROBOFLOW_API_KEY = "<your key>"
.\.venv-ml\Scripts\python.exe tools\ml\audit_with_roboflow.py raw_static_single `
  --num-hands 1 --output roboflow_audit.csv
```

The key is read only from `ROBOFLOW_API_KEY`. The tool writes a report outside
the dataset folder and never relabels or modifies source media; mismatches
are for human review.

## Tests

```powershell
.\.venv-ml\Scripts\python.exe -m pytest tools\ml\tests
```

Tests use synthetic CSVs only. The TFLite contract test is skipped when
TensorFlow is not installed.

## Outputs

Each trainer produces a Keras model, verified TFLite model, model metadata,
training history, confusion matrix, and `mobile_bundle/`. TFLite export passes
only when its verification prediction matches the Keras model prediction.
