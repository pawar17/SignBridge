"""
Landmark-based MLP classifier for sign language recognition.
Input: 63 normalized hand landmark coordinates (21 points × x,y,z)
Output: sign class probabilities
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json


class LandmarkMLP(nn.Module):
    """
    Multi-layer perceptron for hand landmark classification.
    Robust to lighting, background, and hand scale variations.
    """

    def __init__(self, input_dim: int = 63, hidden_dims: list = None, num_classes: int = 26, dropout: float = 0.3):
        super(LandmarkMLP, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h

        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def normalize_landmarks(landmarks_63: "np.ndarray") -> "np.ndarray":
    """
    Normalize hand landmarks to be invariant to position and scale.

    Strategy:
      1. Re-centre on wrist (landmark 0)
      2. Scale by the distance from wrist to middle-finger MCP (landmark 9)

    Args:
        landmarks_63: flat array of 63 floats (21 points × x,y,z)

    Returns:
        Normalized 63-float array, or None if the hand is too small to normalize.
    """
    import numpy as np

    lm = landmarks_63.reshape(21, 3).astype(np.float32)

    wrist = lm[0].copy()
    lm -= wrist                                   # centre on wrist

    scale = float(np.linalg.norm(lm[9]))          # wrist-to-middle-MCP distance
    if scale < 1e-6:
        return None                               # degenerate detection

    lm /= scale
    return lm.flatten()


def get_model(num_classes: int = 26, checkpoint_path=None, device=None):
    """
    Instantiate (and optionally load) a LandmarkMLP.

    Returns (model, label_mapping) where label_mapping may be None.
    """
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LandmarkMLP(num_classes=num_classes)

    label_mapping = None
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        label_mapping = ckpt.get("label_mapping")
        print(f"Loaded landmark model from {checkpoint_path}  "
              f"(val_acc={ckpt.get('val_acc', '?'):.4f})")

    model = model.to(device)
    model.eval()
    return model, label_mapping


if __name__ == "__main__":
    import torch
    model = LandmarkMLP(num_classes=26)
    total = sum(p.numel() for p in model.parameters())
    print(f"LandmarkMLP  params={total:,}")
    x = torch.randn(8, 63)
    out = model(x)
    print(f"Input {x.shape} → Output {out.shape}")
