"""
Extract two-hand MediaPipe landmarks from the ISL alphabet image dataset
(ayesha-hannure/Indian-Sign-Language-dataset, Apache-2.0).

Uses the same hand_landmarker.task model the browser app runs, so training
features match live webcam features. Images are 100x100, so they're upscaled
before detection. Output: one JSON line per image with up to two hands.

    python scripts/training/extract_isl_landmarks.py <dataset_dir> <out.jsonl>
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image

MODEL = Path(__file__).resolve().parents[2] / "web" / "public" / "models" / "hand_landmarker.task"
_landmarker = None


def _init():
    global _landmarker
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )
    _landmarker = vision.HandLandmarker.create_from_options(opts)


def _extract(path):
    img = Image.open(path).convert("RGB").resize((320, 320), Image.BICUBIC)
    res = _landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img)))
    hands = [[[round(p.x, 5), round(p.y, 5)] for p in lm] for lm in res.hand_landmarks]
    return {"label": Path(path).parent.name.upper(), "file": Path(path).name, "hands": hands}


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    files = sorted(str(p) for p in src.glob("*/*.jpg"))
    with Pool(8, initializer=_init) as pool, open(out, "w") as f:
        for i, row in enumerate(pool.imap_unordered(_extract, files, chunksize=32)):
            f.write(json.dumps(row) + "\n")
            if i % 2000 == 0:
                print(i, "/", len(files), flush=True)
