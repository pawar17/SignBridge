# SignBridge

Fingerspelling translation between sign and speech, and between sign languages, running entirely in the browser.

**Live demo: https://pawar17.github.io/SignBridge/**

| Mode | What it does |
| --- | --- |
| Sign → Text | Fingerspell to the camera. SignBridge reads each handshape, types it, and can speak the result aloud. |
| Speech → Sign | Say or type a word or name. It plays back as a fingerspelled sequence of real handshapes. |
| Sign → Sign | Fingerspell in one language (ASL, ISL or BSL) and see the same letters in another. |
| Teach | Record your own version of any letter. Recognition adapts to your hands and camera. |

The camera view draws the 21-point MediaPipe hand skeleton live, so you can see exactly what the model is reading. Video never leaves the device.

## How it works

1. **Hand tracking.** MediaPipe Hand Landmarker (WASM, served with the app) finds up to two hands and 21 landmarks per hand, every frame.
2. **Handshape features.** Each hand becomes 42 numbers that don't depend on where the hand is, how far it is from the camera, or which hand it is: 15 joint-bend angles, fingertip-to-fingertip and fingertip-to-wrist distances scaled by palm size, hand tilt, and fingertip positions in the palm's own frame. Two-handed alphabets (ISL, BSL) add the second hand and the offset between them (87 numbers). Training data is mirrored so left- and right-handed signers both work.
3. **Classifier.** A small neural network (MLP 128-64) trained in scikit-learn and exported to JSON. The browser runs the forward pass itself, so there's no server and no ML runtime to download.
4. **Typing.** Predictions are smoothed over a short window. A letter is typed once it's held steady (~0.65 s); dropping your hand adds a space.
5. **Personalization.** Letters you teach are stored in your browser and matched with nearest-neighbour search. Near a taught sample, your examples outweigh the base model.

The features are computed identically in Python (`scripts/training/train_browser_models.py`) and TypeScript (`web/src/lib/features.ts`). Running the exported model through the browser code reproduces the training predictions (99.6% agreement on the ASL landmark set).

## Accuracy, honestly

| Language | Data | Held-out accuracy | Notes |
| --- | --- | --- | --- |
| ASL | 44,735 samples, two public datasets | 92.7% | Trained on one dataset and tested on the other (new signers): 49%. Teaching a few letters closes most of that gap. J and Z involve motion; the static handshape is used. |
| ISL | 3,473 samples (from 12,637 images where a hand was detected) | 73.3% | Beta. The source images are 100×100, so landmark detection is noisy. |
| BSL | None usable | n/a | No public BSL fingerspelling landmark dataset worked, so BSL runs entirely on letters you teach. |

Held-out accuracy is measured on a random 15% split. The cross-dataset number is the better guide to how it does for a new person on a new camera, which is why Teach exists.

What this does **not** do: full sign language translation. Signed languages use movement, facial grammar and space; this reads fingerspelled letters only.

## Data and licenses

- ASL: [AgEnt-F0X/ASL-Recognition-System](https://github.com/AgEnt-F0X/ASL-Recognition-System) keypoints (MIT) and [lmohammedaariff/Real-Time-ASL-Alphabet-Recognition-Using-Normalized-Hand-Landmarks](https://github.com/lmohammedaariff/Real-Time-ASL-Alphabet-Recognition-Using-Normalized-Hand-Landmarks) landmarks
- ISL: [ayesha-hannure/Indian-Sign-Language-dataset](https://github.com/ayesha-hannure/Indian-Sign-Language-dataset) (Apache-2.0), landmarks extracted with `scripts/training/extract_isl_landmarks.py`
- Hand tracking: [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) (Apache-2.0)

## Run locally

```bash
cd web
npm install
npm run dev
```

Retrain the models (writes to `web/public/models`):

```bash
pip install scikit-learn numpy mediapipe
python scripts/training/extract_isl_landmarks.py <isl_dataset_dir> isl_landmarks.jsonl
python scripts/training/train_browser_models.py --asl-kp keypoint.csv --asl-lm landmarks.csv \
    --isl isl_landmarks.jsonl --out web/public/models
```

## Repository

- `web/`: the app (Vite, React, TypeScript)
- `scripts/training/`: landmark extraction and model training
- `backend/`, `frontend/`, `models/`, `notebooks/`: the earlier server-based prototype, kept for reference ([old README](docs/README_v1.md))

Built by [Aadya Pawar](https://aadyapawar.com).
