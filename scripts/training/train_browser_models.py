"""
Train the landmark classifiers the SignBridge browser app runs.

Features match what the browser computes from MediaPipe Hand Landmarker
(see hand_features below and web/src/lib/features.ts): joint angles, palm-scaled
distances and palm-frame fingertip positions, so position, distance from the
camera and signer don't matter much. Every training sample is also mirrored,
so either hand works.

Data (all public):
  ASL  kinivi-format keypoints, AgEnt-F0X/ASL-Recognition-System (MIT), 36k samples
       landmarks.csv, lmohammedaariff/Real-Time-ASL-Alphabet-Recognition... (held out
       as a separate-signer test set, then merged in for the final model)
  ISL  two-hand landmarks extracted by extract_isl_landmarks.py from
       ayesha-hannure/Indian-Sign-Language-dataset (Apache-2.0)

    python scripts/training/train_browser_models.py --asl-kp keypoint.csv \
        --asl-lm landmarks.csv --isl isl_landmarks.jsonl --out web/public/models
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

LETTERS = [chr(ord("A") + i) for i in range(26)]


CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
TIPS = [4, 8, 12, 16, 20]


def hand_features(P):
    """
    P: (21, 2) landmark points in square units (x and y on the same scale).
    -> 42 values describing the handshape, robust to position, distance and signer:
       15 joint-bend cosines, 10 fingertip-to-fingertip distances, 5 fingertip-to-wrist
       distances (all scaled by palm size), the hand's tilt (sin, cos), and the five
       fingertips in the palm's own frame. The browser computes the identical features
       (web/src/lib/features.ts).
    """
    rel = P - P[0]
    palm = np.linalg.norm(rel[9]) + 1e-6
    f = []
    for ch in CHAINS:
        for i in range(1, 4):
            a = P[ch[i - 1]] - P[ch[i]]
            b = P[ch[i + 1]] - P[ch[i]]
            f.append(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
    for i in range(5):
        for j in range(i + 1, 5):
            f.append(np.linalg.norm(P[TIPS[i]] - P[TIPS[j]]) / palm)
    for t in TIPS:
        f.append(np.linalg.norm(P[t] - P[0]) / palm)
    ang = np.arctan2(rel[9, 0], -rel[9, 1])
    f += [np.sin(ang), np.cos(ang)]
    c, s = np.cos(-ang), np.sin(-ang)
    R = rel @ np.array([[c, -s], [s, c]]).T / palm
    f += list(R[TIPS].ravel())
    return np.array(f)


def mirror(P):
    Q = P.copy()
    Q[:, 0] = -Q[:, 0]
    return Q


def two_hand_features(hands):
    """hands: list of (21,2) arrays (1 or 2) -> 87 values: hand 1, hand 2, wrist offset, two-hand flag."""
    hands = sorted(hands, key=lambda h: h[0, 0])  # left-most hand in the image first
    h1 = hand_features(hands[0])
    if len(hands) == 1:
        return np.concatenate([h1, np.zeros(42), np.zeros(2), [0.0]])
    h2 = hand_features(hands[1])
    palm = (np.linalg.norm(hands[0][9] - hands[0][0]) + np.linalg.norm(hands[1][9] - hands[1][0])) / 2 + 1e-6
    off = np.clip((hands[1][0] - hands[0][0]) / palm, -6, 6)
    return np.concatenate([h1, h2, off, [1.0]])


def pose_for_display(P):
    """Wrist-relative points scaled by palm size, for drawing reference handshapes."""
    rel = P - P[0]
    return rel / (np.linalg.norm(rel[9]) + 1e-6)


def load_asl_kp(path):
    """kinivi-format rows: label index, then 21 (x, y) wrist-relative pixel offsets."""
    P, y = [], []
    for row in csv.reader(open(path, encoding="utf-8-sig")):
        P.append(np.array([float(t) for t in row[1:43]]).reshape(21, 2))
        y.append(LETTERS[int(row[0])])
    return np.array(P), np.array(y)


def load_asl_lm(path):
    """63 values per row (21 x, y, z, wrist-relative, scaled); keep x, y."""
    P, y = [], []
    rows = csv.reader(open(path))
    next(rows)
    for row in rows:
        P.append(np.array([float(t) for t in row[:63]]).reshape(21, 3)[:, :2])
        y.append(row[63])
    return np.array(P), np.array(y)


def load_isl(path):
    """Two-hand samples (images were square, so normalized x, y share a scale)."""
    samples, y = [], []
    for line in open(path):
        r = json.loads(line)
        if r["hands"]:
            samples.append([np.array(h) for h in r["hands"][:2]])
            y.append(r["label"])
    return samples, np.array(y)


def one_hand_xy(P, y):
    X = np.array([hand_features(p) for p in P] + [hand_features(mirror(p)) for p in P])
    return X, np.concatenate([y, y])


def two_hand_xy(samples, y):
    X = [two_hand_features(h) for h in samples] + [two_hand_features([mirror(p) for p in h]) for h in samples]
    return np.array(X), np.concatenate([y, y])


def medoid_poses(P, y, feats):
    """For each letter, the real sample closest to the class's average features."""
    out = {}
    for label in sorted(set(y)):
        idx = np.where(y == label)[0]
        centre = feats[idx].mean(0)
        best = idx[np.argmin(np.linalg.norm(feats[idx] - centre, axis=1))]
        out[label] = best
    return out


class Model:
    """MLP on standardized features; labels kept as strings outside sklearn."""

    def __init__(self, scaler, clf, labels):
        self.scaler, self.clf, self.labels = scaler, clf, labels

    def predict(self, X):
        return self.labels[self.clf.predict(self.scaler.transform(X))]


def fit(X, y, hidden):
    labels = np.array(sorted(set(y)))
    yi = np.searchsorted(labels, y)
    scaler = StandardScaler().fit(X)
    clf = MLPClassifier(hidden_layer_sizes=hidden, alpha=1e-3, max_iter=300, early_stopping=True,
                        n_iter_no_change=15, random_state=7)
    clf.fit(scaler.transform(X), yi)
    return Model(scaler, clf, labels)


def export(path, name, m, features, notes):
    scaler, clf = m.scaler, m.clf
    model = {
        "language": name,
        "features": features,
        "labels": [str(c) for c in m.labels],
        "mean": scaler.mean_.round(6).tolist(),
        "scale": scaler.scale_.round(6).tolist(),
        "layers": [
            {"W": W.T.round(5).tolist(), "b": b.round(5).tolist()}  # W: out x in
            for W, b in zip(clf.coefs_, clf.intercepts_)
        ],
        "notes": notes,
    }
    path.write_text(json.dumps(model, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asl-kp", required=True)
    ap.add_argument("--asl-lm", required=True)
    ap.add_argument("--isl", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    poses = {}

    # ---- ASL (one hand) ----
    Pk, yk = load_asl_kp(a.asl_kp)
    Pl, yl = load_asl_lm(a.asl_lm)
    print("ASL samples:", len(Pk), "+", len(Pl))

    # Separate-signer check: train on dataset A only, test on dataset B's different signers
    m = fit(*one_hand_xy(Pk, yk), (128, 64))
    cross = accuracy_score(yl, m.predict(np.array([hand_features(p) for p in Pl])))
    print(f"ASL cross-dataset accuracy (train A, test B): {cross:.3f}")

    P = np.concatenate([Pk, Pl])
    y = np.concatenate([yk, yl])
    tr, te = train_test_split(np.arange(len(y)), test_size=0.15, stratify=y, random_state=7)
    m = fit(*one_hand_xy(P[tr], y[tr]), (128, 64))
    Xte = np.array([hand_features(p) for p in P[te]])
    hold = accuracy_score(y[te], m.predict(Xte))
    print(f"ASL holdout accuracy: {hold:.3f}")
    print(classification_report(y[te], m.predict(Xte), digits=3))

    m = fit(*one_hand_xy(P, y), (128, 64))
    export(out / "asl.json", "ASL", m, "one-hand-42",
           {"holdout_accuracy": round(hold, 4), "cross_dataset_accuracy": round(cross, 4), "samples": int(len(y))})
    feats = np.array([hand_features(p) for p in Pl])
    med = medoid_poses(Pl, yl, feats)  # display poses from the multi-signer photo set
    poses["ASL"] = {k: {"hands": [pose_for_display(Pl[i]).round(3).tolist()]} for k, i in med.items()}
    report["ASL"] = {"holdout": round(hold, 4), "cross_dataset": round(cross, 4), "samples": int(len(y))}

    # ---- ISL (one or two hands) ----
    samples, yi = load_isl(a.isl)
    print("ISL samples:", len(samples))
    tr, te = train_test_split(np.arange(len(yi)), test_size=0.15, stratify=yi, random_state=7)
    m = fit(*two_hand_xy([samples[i] for i in tr], yi[tr]), (128, 64))
    Xte = np.array([two_hand_features(samples[i]) for i in te])
    hold = accuracy_score(yi[te], m.predict(Xte))
    print(f"ISL holdout accuracy: {hold:.3f}")
    print(classification_report(yi[te], m.predict(Xte), digits=3))
    m = fit(*two_hand_xy(samples, yi), (128, 64))
    export(out / "isl.json", "ISL", m, "two-hand-87", {"holdout_accuracy": round(hold, 4), "samples": int(len(yi))})
    report["ISL"] = {"holdout": round(hold, 4), "samples": int(len(yi))}

    # ISL display poses: prefer two-hand samples; both hands in one frame, scaled by palm size
    isl_poses = {}
    for label in sorted(set(yi)):
        idx = [i for i in np.where(yi == label)[0] if len(samples[i]) == 2] or list(np.where(yi == label)[0])
        F = np.array([two_hand_features(samples[i]) for i in idx])
        best = idx[int(np.argmin(np.linalg.norm(F - F.mean(0), axis=1)))]
        hs = sorted(samples[best], key=lambda h: h[0, 0])
        palm = np.mean([np.linalg.norm(h[9] - h[0]) for h in hs]) + 1e-6
        centre = np.mean([h[0] for h in hs], axis=0)
        isl_poses[label] = {"hands": [((h - centre) / palm).round(3).tolist() for h in hs]}
    poses["ISL"] = isl_poses

    (out / "poses.json").write_text(json.dumps(poses, separators=(",", ":")))
    (out / "training_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
