"""
Extract MediaPipe hand landmarks from image datasets and train a per-language
MLP classifier that works robustly on real webcam input.

Supported languages (by available training data):
  - ASL  -> data/raw/asl/custom_original/asl_dataset/   (a-z, 0-9, ~70 imgs/class)
  - ISL  -> data/raw/isl/numbers_and_letters/            (A-Z + 0-9, ~1200 imgs/class)
           + data/raw/isl/custom/ISL custom Data/        (A-Z, ~550 imgs/class)
  - BSL  -> no data; falls back to ASL model at runtime

Usage (from project root):
    python scripts/training/extract_and_train_landmarks.py --lang ASL
    python scripts/training/extract_and_train_landmarks.py --lang ISL
    python scripts/training/extract_and_train_landmarks.py --lang ALL
"""

import sys
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# ── project paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from models.landmark_mlp import LandmarkMLP, normalize_landmarks

CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = ROOT / "data" / "raw"

# ── MediaPipe (Tasks API — mediapipe >= 0.10) ─────────────────────────────────
MEDIAPIPE_OK = False
_mp_vision   = None
_mp_module   = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python as _mp_tasks_python
    from mediapipe.tasks.python import vision as _mp_vision_mod
    _mp_module  = mp
    _mp_vision  = _mp_vision_mod
    MEDIAPIPE_OK = True
except ImportError:
    print("WARNING: mediapipe not installed  -> pip install mediapipe")

MODEL_PATH = ROOT / "models" / "mediapipe" / "hand_landmarker.task"

def _ensure_model():
    """Download hand landmarker model if needed."""
    if MODEL_PATH.exists():
        return True
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    url = ("https://storage.googleapis.com/mediapipe-models/"
           "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    print(f"Downloading MediaPipe hand-landmarker model (~8 MB)…")
    import urllib.request
    urllib.request.urlretrieve(url, MODEL_PATH)
    print(f"Saved ->{MODEL_PATH}")
    return True

def _make_detector():
    """Create a fresh HandLandmarker for static-image mode."""
    _ensure_model()
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    base = mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset definitions
# Each entry: (root_dir, case_transform)
#   root_dir       – folder whose immediate children are class-name sub-dirs
#   case_transform – 'upper' | 'lower' | None
# ─────────────────────────────────────────────────────────────────────────────
DATASETS = {
    "ASL": [
        (DATA_ROOT / "asl" / "custom_original" / "asl_dataset", "upper"),
    ],
    "ISL": [
        (DATA_ROOT / "isl" / "numbers_and_letters",           "upper"),
        (DATA_ROOT / "isl" / "custom" / "ISL custom Data",   "upper"),
    ],
}

# letters-only filter for sign language (drop digits for fingerspelling model)
LETTER_ONLY = False   # set True to keep only A-Z classes


# ─────────────────────────────────────────────────────────────────────────────
# Landmark extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_landmarks_from_image(img_bgr, detector) -> np.ndarray | None:
    """
    Run MediaPipe Tasks HandLandmarker on a BGR image.
    Returns normalized 63-float vector or None if no hand detected.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img  = _mp_module.Image(image_format=_mp_module.ImageFormat.SRGB, data=img_rgb)
    result  = detector.detect(mp_img)

    if not result.hand_landmarks:
        return None

    lm  = result.hand_landmarks[0]
    raw = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()
    return normalize_landmarks(raw)


def load_dataset_landmarks(lang: str, verbose: bool = True):
    """
    Walk every image in the language dataset dirs, extract landmarks,
    and return (X, y, label_to_idx).

    X : float32 array (N, 63)
    y : int64 array  (N,)
    label_to_idx : dict{str ->int}
    """
    assert MEDIAPIPE_OK, "mediapipe is required for landmark extraction"

    sources = DATASETS[lang]
    # First pass: collect all valid class names
    class_names = set()
    for root, case in sources:
        root = Path(root)
        if not root.exists():
            print(f"  [SKIP] {root}  (not found)")
            continue
        for cls_dir in sorted(root.iterdir()):
            if cls_dir.is_dir():
                name = cls_dir.name.upper() if case == "upper" else cls_dir.name.lower()
                if LETTER_ONLY and not name.isalpha():
                    continue
                class_names.add(name)

    class_names = sorted(class_names)
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    if verbose:
        print(f"\n{lang}: {len(class_names)} classes -> {class_names}")

    X_list, y_list = [], []
    skipped = 0

    detector = _make_detector()

    for root, case in sources:
        root = Path(root)
        if not root.exists():
            continue
        for cls_dir in sorted(root.iterdir()):
            if not cls_dir.is_dir():
                continue
            name = cls_dir.name.upper() if case == "upper" else cls_dir.name.lower()
            if name not in label_to_idx:
                continue
            label_idx = label_to_idx[name]

            imgs = [p for p in cls_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

            ok, bad = 0, 0
            for img_path in imgs:
                img = cv2.imread(str(img_path))
                if img is None:
                    bad += 1
                    continue
                feat = extract_landmarks_from_image(img, detector)
                if feat is None:
                    bad += 1
                    skipped += 1
                    continue
                X_list.append(feat)
                y_list.append(label_idx)
                ok += 1

            if verbose:
                print(f"  {name:3s} ->{ok:4d} ok  {bad:3d} skipped")

    detector.close()

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    print(f"\n{lang}: extracted {len(X)} samples "
          f"({skipped} skipped – no hand detected)")
    return X, y, label_to_idx

# ─────────────────────────────────────────────────────────────────────────────
# Training & Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment_landmarks(Xb: torch.Tensor, jitter_std: float = 0.015) -> torch.Tensor:
    """
    Data Augmentation: Add random Gaussian noise to the landmark coordinates.
    This simulates camera shake, motion blur, or MediaPipe tracking jitter,
    making the MLP significantly more robust to imperfect real-world inputs.
    """
    noise = torch.randn_like(Xb) * jitter_std
    return Xb + noise


def train_model(lang: str,
                epochs: int = 60,
                batch_size: int = 128,
                lr: float = 1e-3,
                val_split: float = 0.15):
    print(f"\n{'='*60}")
    print(f"  Training landmark MLP for {lang}")
    print(f"{'='*60}")

    X, y, label_to_idx = load_dataset_landmarks(lang, verbose=True)
    num_classes = len(label_to_idx)

    # ── tensors ──────────────────────────────────────────────────────────────
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    dataset = TensorDataset(X_t, y_t)

    val_size  = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=256,        shuffle=False,
                              num_workers=0, pin_memory=False)

    # ── model ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = LandmarkMLP(
        input_dim=63,
        hidden_dims=[512, 256, 128],
        num_classes=num_classes,
        dropout=0.3,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state   = None

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            
            # Apply landmark augmentation during training
            Xb_aug = augment_landmarks(Xb)
            
            optimizer.zero_grad()
            logits = model(Xb_aug)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(yb)
            correct    += (logits.argmax(1) == yb).sum().item()
            total      += len(yb)

        scheduler.step()

        # ── validation ───────────────────────────────────────────────────────
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds  = model(Xb).argmax(1)
                val_correct += (preds == yb).sum().item()
                val_total   += len(yb)

        train_acc = correct     / total
        val_acc   = val_correct / val_total
        avg_loss  = total_loss  / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"loss={avg_loss:.4f}  "
                  f"train_acc={train_acc:.4f}  "
                  f"val_acc={val_acc:.4f}  "
                  f"best_val={best_val_acc:.4f}")

    # ── save ─────────────────────────────────────────────────────────────────
    out_path = CHECKPOINT_DIR / f"{lang.lower()}_landmark_model.pth"
    torch.save({
        "model_state_dict": best_state,
        "num_classes":      num_classes,
        "label_to_idx":     label_to_idx,
        "idx_to_label":     {str(i): c for c, i in label_to_idx.items()},
        "val_acc":          best_val_acc,
        "input_dim":        63,
        "hidden_dims":      [512, 256, 128],
        "label_mapping": {
            "label_to_idx": label_to_idx,
            "idx_to_label": {str(i): c for c, i in label_to_idx.items()},
        },
    }, out_path)
    print(f"\nSaved ->{out_path}  (best val acc = {best_val_acc:.4f})")

    # ── also save label mapping as JSON (for backend) ────────────────────────
    mapping_path = CHECKPOINT_DIR / f"{lang.lower()}_label_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump({
            "language":     lang,
            "num_classes":  num_classes,
            "label_to_idx": label_to_idx,
            "idx_to_label": {str(i): c for c, i in label_to_idx.items()},
        }, f, indent=2)
    print(f"Saved ->{mapping_path}")

    return best_val_acc


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract landmarks and train sign-language MLP(s)")
    parser.add_argument("--lang", type=str, default="ALL",
                        choices=["ASL", "ISL", "ALL"],
                        help="Language to train (default: ALL)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch",  type=int, default=128)
    parser.add_argument("--lr",     type=float, default=1e-3)
    args = parser.parse_args()

    langs = list(DATASETS.keys()) if args.lang == "ALL" else [args.lang]

    results = {}
    t0 = time.time()
    for lang in langs:
        try:
            acc = train_model(lang, epochs=args.epochs,
                              batch_size=args.batch, lr=args.lr)
            results[lang] = acc
        except Exception as e:
            print(f"ERROR training {lang}: {e}")
            import traceback; traceback.print_exc()
            results[lang] = None

    print(f"\n{'='*60}")
    print(f"Training complete in {(time.time()-t0)/60:.1f} min")
    for lang, acc in results.items():
        status = f"val_acc={acc:.4f}" if acc is not None else "FAILED"
        print(f"  {lang}: {status}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
