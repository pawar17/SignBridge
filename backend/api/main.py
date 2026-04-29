"""
SignBridge FastAPI Application — v2 (landmark-based inference + ElevenLabs TTS)

Architecture:
  - MediaPipe Hands  → 21 hand landmarks (x,y,z)  → 63 features
  - Normalize relative to wrist + hand scale
  - LandmarkMLP  → class probabilities
  - Per-language models: ASL, ISL (BSL falls back to ASL)
  - ElevenLabs TTS  → Sign → Speech audio

Endpoints:
  GET  /              health / status
  GET  /health
  GET  /languages
  GET  /classes?lang=ASL
  POST /predict           form-data: file + lang
  POST /predict_sentence  form-data: file + lang + session_id
  POST /clear_sentence/{session_id}
  POST /tts               form-data: text + voice_id (optional)
  GET  /tts/voices        list available ElevenLabs voices
  GET  /learn/{letter}
  POST /practice/{letter}
"""

from contextlib import asynccontextmanager
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import io
import json
import os
import sys

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, Header, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from models.landmark_mlp import LandmarkMLP, normalize_landmarks

# ── constants ─────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
LANG_FALLBACK   = {"BSL": "ASL"}  # BSL has no training data → use ASL model
SUPPORTED_LANGS  = ["ASL", "BSL", "ISL"]

# ─────────────────────────────────────────────────────────────────────────────
# Language-model registry
# ─────────────────────────────────────────────────────────────────────────────

class LangModelRegistry:
    """Holds one LandmarkMLP per language, loaded from checkpoints."""

    def __init__(self):
        self.models: dict[str, LandmarkMLP] = {}
        self.label_maps: dict[str, dict] = {}   # lang → {idx_to_label, label_to_idx}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self, lang: str, path: Path) -> bool:
        try:
            ckpt = torch.load(path, map_location=self.device)
            n    = ckpt["num_classes"]
            hdim = ckpt.get("hidden_dims", [512, 256, 128])
            m    = LandmarkMLP(input_dim=63, hidden_dims=hdim, num_classes=n)
            m.load_state_dict(ckpt["model_state_dict"])
            m.to(self.device).eval()
            self.models[lang]    = m
            self.label_maps[lang] = {
                "idx_to_label": ckpt.get("idx_to_label", {}),
                "label_to_idx": ckpt.get("label_to_idx", {}),
                "num_classes":  n,
            }
            acc = ckpt.get("val_acc", 0.0)
            print(f"  [{lang}] loaded {n} classes  val_acc={acc:.4f}  {path.name}")
            return True
        except Exception as e:
            print(f"  [{lang}] FAILED to load {path}: {e}")
            return False

    def load_all(self):
        print(f"\nLoading landmark models from {CHECKPOINT_DIR}")
        for lang in SUPPORTED_LANGS:
            effective = LANG_FALLBACK.get(lang, lang)
            path = CHECKPOINT_DIR / f"{effective.lower()}_landmark_model.pth"
            if path.exists():
                if effective not in self.models:          # don't load twice
                    self.load(effective, path)
                if effective != lang:
                    # alias: BSL → ASL
                    self.models[lang]    = self.models[effective]
                    self.label_maps[lang] = self.label_maps[effective]
            else:
                print(f"  [{lang}] checkpoint not found: {path}")
        if not self.models:
            print("  WARNING: no landmark models loaded — run training script first")

    def predict(self, lang: str, feat63: np.ndarray) -> dict:
        effective = LANG_FALLBACK.get(lang, lang)
        m = self.models.get(effective) or next(iter(self.models.values()), None)
        if m is None:
            raise RuntimeError("No model available")

        lmap = self.label_maps.get(effective) or next(iter(self.label_maps.values()))

        t = torch.from_numpy(feat63).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = m(t)
            probs  = torch.softmax(logits, dim=1)[0]

        top3_v, top3_i = probs.topk(min(3, len(probs)))
        idx2lbl = lmap["idx_to_label"]

        def lbl(idx): return idx2lbl.get(str(idx), idx2lbl.get(idx, str(idx)))

        best_idx  = top3_i[0].item()
        best_conf = top3_v[0].item()
        return {
            "prediction": lbl(best_idx),
            "confidence": best_conf,
            "top_3": [{"class": lbl(i.item()), "confidence": v.item()}
                      for v, i in zip(top3_v, top3_i)],
        }

    def available_langs(self):
        return sorted(self.models.keys())


registry = LangModelRegistry()


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe (Tasks API — mediapipe >= 0.10)
# ─────────────────────────────────────────────────────────────────────────────

MEDIAPIPE_OK    = False
_mp_module      = None
_hands_detector = None

MODEL_FILE = ROOT / "models" / "mediapipe" / "hand_landmarker.task"


def _ensure_mp_model():
    if MODEL_FILE.exists():
        return True
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    url = ("https://storage.googleapis.com/mediapipe-models/"
           "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    print("Downloading MediaPipe hand-landmarker model …")
    urllib.request.urlretrieve(url, MODEL_FILE)
    print(f"Saved → {MODEL_FILE}")
    return True


def _init_mediapipe():
    global MEDIAPIPE_OK, _mp_module, _hands_detector
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        _ensure_mp_model()
        _mp_module = mp

        base = mp_tasks.BaseOptions(model_asset_path=str(MODEL_FILE))
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_score=0.5,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        _hands_detector = mp_vision.HandLandmarker.create_from_options(opts)
        MEDIAPIPE_OK = True
        print("MediaPipe HandLandmarker ready (Tasks API)")
    except Exception as e:
        print(f"MediaPipe init failed: {e}")
        MEDIAPIPE_OK = False

_init_mediapipe()


def extract_landmarks_from_pil(pil_image: Image.Image) -> np.ndarray | None:
    """
    Run MediaPipe Tasks HandLandmarker on a PIL image.
    Returns normalized 63-float numpy array or None if no hand found.
    """
    if not MEDIAPIPE_OK or _hands_detector is None:
        return None

    img_rgb = np.array(pil_image.convert("RGB"))
    mp_img  = _mp_module.Image(image_format=_mp_module.ImageFormat.SRGB, data=img_rgb)
    result  = _hands_detector.detect(mp_img)

    if not result.hand_landmarks:
        return None

    lm  = result.hand_landmarks[0]
    raw = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()
    return normalize_landmarks(raw)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.load_all()
    yield

app = FastAPI(
    title="SignBridge API",
    description="Landmark-based multi-language sign recognition",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Sentence builder
#
# State per session:
#   current_letter   – letter currently being held (for dedup)
#   letter_frames    – how many consecutive frames showing current_letter
#   current_word     – letters accumulated into the word being signed
#   words            – list of completed words (sentence so far)
#   last_hand_time   – last time a hand was detected
#   last_letter_time – last time a new letter was appended
#   no_hand_count    – consecutive frames with no hand (for word-break detection)
#
# Rules:
#   • A letter is committed after LETTER_HOLD_FRAMES consecutive identical frames
#   • Holding a letter steady does NOT keep appending it (dedup)
#   • When no hand is detected for WORD_BREAK_FRAMES consecutive frames
#     → current word ends, a space is inserted
#   • When no hand for SESSION_RESET_SECS the whole sentence resets
# ─────────────────────────────────────────────────────────────────────────────

LETTER_HOLD_FRAMES  = 3    # frames a letter must be stable before committing
WORD_BREAK_FRAMES   = 8    # consecutive no-hand frames → end of word (≈ 2-3 s at 3 fps)
SESSION_RESET_SECS  = 30   # seconds of total inactivity → full sentence reset

def _new_session():
    return {
        "current_letter":   None,
        "letter_frames":    0,
        "current_word":     [],    # list of committed letters in current word
        "words":            [],    # list of completed word strings
        "last_hand_time":   None,
        "last_letter_time": datetime.now(),
        "no_hand_count":    0,
    }

sentence_sessions: dict = defaultdict(_new_session)


def _build_sentence(sess: dict) -> str:
    """Reconstruct the sentence from completed words + current partial word."""
    parts = list(sess["words"])
    if sess["current_word"]:
        parts.append("".join(sess["current_word"]))
    return " ".join(parts)


def _sentence_step(session_id: str, letter: str | None, conf: float) -> dict:
    """
    Feed one frame into the sentence builder.
    letter=None means no hand detected this frame.
    Returns updated sentence state.
    """
    sess = sentence_sessions[session_id]
    now  = datetime.now()

    # ── full reset after long silence ─────────────────────────────────────
    if sess["last_letter_time"] and (now - sess["last_letter_time"]).total_seconds() > SESSION_RESET_SECS:
        sentence_sessions[session_id] = _new_session()
        sess = sentence_sessions[session_id]

    # ── no hand detected ──────────────────────────────────────────────────
    if letter is None:
        sess["no_hand_count"] += 1
        sess["current_letter"]  = None
        sess["letter_frames"]   = 0

        if sess["no_hand_count"] >= WORD_BREAK_FRAMES and sess["current_word"]:
            # commit current word → move to words list
            sess["words"].append("".join(sess["current_word"]))
            sess["current_word"] = []

        return {
            "committed": False,
            "letter":    None,
            "sentence":  _build_sentence(sess),
            "words":     list(sess["words"]),
            "word":      "".join(sess["current_word"]),
        }

    # ── hand detected ─────────────────────────────────────────────────────
    sess["no_hand_count"]  = 0
    sess["last_hand_time"] = now

    if letter == sess["current_letter"]:
        sess["letter_frames"] += 1
    else:
        sess["current_letter"] = letter
        sess["letter_frames"]  = 1

    committed = False
    if sess["letter_frames"] == LETTER_HOLD_FRAMES:
        # stable for enough frames → commit this letter
        sess["current_word"].append(letter)
        sess["last_letter_time"] = now
        committed = True

    return {
        "committed": committed,
        "letter":    letter,
        "sentence":  _build_sentence(sess),
        "words":     list(sess["words"]),
        "word":      "".join(sess["current_word"]),
        "frames_held": sess["letter_frames"],
    }

# rolling smoothing window — last N raw predictions per camera session
# Using a simple deque-based majority vote for stability
from collections import deque
_smooth_window: dict = defaultdict(lambda: deque(maxlen=5))   # last 5 frames

def _smooth_prediction(session_key: str, pred: str, conf: float) -> tuple[str, float]:
    """
    Simple temporal smoothing: keep last 5 predictions and return the
    most frequent one.  If the current frame is very high confidence
    (> 0.85) we trust it immediately.
    """
    if conf > 0.85:
        _smooth_window[session_key].clear()
        _smooth_window[session_key].append(pred)
        return pred, conf

    _smooth_window[session_key].append(pred)
    window = list(_smooth_window[session_key])

    # majority vote
    from collections import Counter
    vote, count = Counter(window).most_common(1)[0]
    smoothed_conf = conf * (count / len(window))   # scale conf by agreement
    return vote, smoothed_conf

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    top_3: list
    success: bool
    hand_detected: bool
    lang: str


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "SignBridge API v2 (landmark-based)",
        "version": "2.0.0",
        "status": "running",
        "models_loaded": registry.available_langs(),
        "mediapipe": MEDIAPIPE_OK,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": registry.available_langs(),
        "mediapipe": MEDIAPIPE_OK,
    }


@app.get("/languages")
async def get_languages():
    """Available language pairs for Sign-to-Text and Sign-to-Sign modes."""
    return {
        "sign_to_text": [
            {"id": "ASL-EN",  "sign_lang": "ASL", "text_lang": "English", "sign_flag": "🇺🇸", "text_flag": "🇺🇸"},
            {"id": "BSL-EN",  "sign_lang": "BSL", "text_lang": "English", "sign_flag": "🇬🇧", "text_flag": "🇬🇧"},
            {"id": "ISL-HI",  "sign_lang": "ISL", "text_lang": "Hindi",   "sign_flag": "🇮🇳", "text_flag": "🇮🇳"},
            {"id": "ASL-FR",  "sign_lang": "ASL", "text_lang": "French",  "sign_flag": "🇺🇸", "text_flag": "🇫🇷"},
        ],
        "sign_to_sign": [
            {"id": "ASL-BSL", "input": "ASL", "output": "BSL", "in_flag": "🇺🇸", "out_flag": "🇬🇧"},
            {"id": "ASL-ISL", "input": "ASL", "output": "ISL", "in_flag": "🇺🇸", "out_flag": "🇮🇳"},
            {"id": "ISL-ASL", "input": "ISL", "output": "ASL", "in_flag": "🇮🇳", "out_flag": "🇺🇸"},
            {"id": "BSL-ASL", "input": "BSL", "output": "ASL", "in_flag": "🇬🇧", "out_flag": "🇺🇸"},
        ],
    }


@app.get("/classes")
async def get_classes(lang: str = "ASL"):
    lang = lang.upper()
    effective = LANG_FALLBACK.get(lang, lang)
    lmap = registry.label_maps.get(effective)
    if lmap is None:
        # return default ASL letters
        labels = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
        return {"classes": labels, "count": len(labels), "lang": lang, "model": "default"}
    labels = sorted(lmap["label_to_idx"], key=lambda c: lmap["label_to_idx"][c])
    return {"classes": labels, "count": len(labels), "lang": lang, "model": effective}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    lang: str = Form("ASL"),
    session_id: str = Form("default"),
):
    """
    Predict sign letter from a webcam frame.

    Parameters:
      file       – JPEG/PNG image (form-data)
      lang       – ASL | BSL | ISL  (default ASL)
      session_id – used for temporal smoothing across frames (default "default")
    """
    lang = lang.upper()

    if not registry.models and not registry.label_maps:
        raise HTTPException(
            status_code=503,
            detail="No models loaded. Run: python scripts/training/extract_and_train_landmarks.py"
        )

    try:
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents))
        feat     = extract_landmarks_from_pil(image)

        if feat is None:
            # no hand detected — clear the smoothing window and return placeholder
            _smooth_window[session_id].clear()
            return PredictionResponse(
                prediction="?",
                confidence=0.0,
                top_3=[],
                success=False,
                hand_detected=False,
                lang=lang,
            )

        result = registry.predict(lang, feat)
        pred   = result["prediction"]
        conf   = result["confidence"]

        # temporal smoothing: average last N frames
        smooth_key = f"{session_id}:{lang}"
        pred, conf = _smooth_prediction(smooth_key, pred, conf)

        return PredictionResponse(
            prediction=pred,
            confidence=conf,
            top_3=result["top_3"],
            success=True,
            hand_detected=True,
            lang=lang,
        )

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_sentence")
async def predict_sentence(
    file: UploadFile = File(...),
    lang: str = Form("ASL"),
    session_id: str = Form("default"),
):
    """
    Predict one frame and update the sentence builder for this session.

    Sentence building rules:
      - A letter must be held for LETTER_HOLD_FRAMES (~3) consecutive stable frames
        before it is committed (prevents jitter-spam).
      - When no hand is detected for WORD_BREAK_FRAMES (~8) frames, the current
        word is closed and a space is inserted.
      - Full sentence resets after SESSION_RESET_SECS of inactivity.

    Returns:
      letter       – current frame prediction ('?' if no hand)
      confidence   – model confidence for this frame
      committed    – whether this frame committed a new letter
      word         – letters in the current in-progress word
      sentence     – full sentence built so far (words separated by spaces)
      frames_held  – how many consecutive frames the current letter has been held
      hand_detected
    """
    lang = lang.upper()

    try:
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents))
        feat     = extract_landmarks_from_pil(image)

        if feat is None:
            # no hand — feed None into sentence builder (may trigger word break)
            _smooth_window[f"{session_id}:{lang}"].clear()
            state = _sentence_step(session_id, None, 0.0)
            return {
                "letter":       "?",
                "confidence":   0.0,
                "committed":    False,
                "word":         state["word"],
                "sentence":     state["sentence"],
                "frames_held":  0,
                "top_3":        [],
                "success":      False,
                "hand_detected": False,
                "lang":         lang,
            }

        result = registry.predict(lang, feat)
        pred   = result["prediction"]
        conf   = result["confidence"]

        # temporal smoothing
        smooth_key   = f"{session_id}:{lang}"
        pred, conf   = _smooth_prediction(smooth_key, pred, conf)

        # sentence builder
        state = _sentence_step(session_id, pred, conf)

        return {
            "letter":       pred,
            "confidence":   conf,
            "committed":    state["committed"],
            "word":         state["word"],
            "sentence":     state["sentence"],
            "frames_held":  state.get("frames_held", 0),
            "top_3":        result["top_3"],
            "success":      True,
            "hand_detected": True,
            "lang":         lang,
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear_sentence/{session_id}")
async def clear_sentence(session_id: str):
    sentence_sessions[session_id] = _new_session()
    _smooth_window[f"{session_id}:ASL"].clear()
    _smooth_window[f"{session_id}:ISL"].clear()
    _smooth_window[f"{session_id}:BSL"].clear()
    return {"status": "cleared", "session_id": session_id}


# ── learning mode ─────────────────────────────────────────────────────────────

LEARNING_CONTENT = {
    "A": {"description": "Closed fist with thumb on the side",
          "tips": ["Keep fingers closed tightly", "Thumb touches the side of fist", "Palm faces forward"],
          "common_mistakes": ["Thumb sticking up", "Fingers not closed", "Fist too loose"]},
    "B": {"description": "Flat hand, fingers together, thumb across palm",
          "tips": ["Four fingers straight and together", "Tuck thumb across palm", "Palm faces forward"],
          "common_mistakes": ["Fingers spread", "Thumb not tucked", "Bent fingers"]},
    "C": {"description": "Curved hand forming letter C",
          "tips": ["Curve all fingers uniformly", "Keep thumb curved", "Show the C shape clearly"],
          "common_mistakes": ["Fingers too straight", "Not curved enough", "Wrong thumb position"]},
    "D": {"description": "Index up, other fingers touch thumb",
          "tips": ["Straight index finger", "M/R/P touch thumb", "Form circle with thumb"],
          "common_mistakes": ["Index bent", "Other fingers not touching", "No circle"]},
    "E": {"description": "Fingers curled down, thumb across",
          "tips": ["Curl all fingers inward", "Thumb across fingers", "Tight position"],
          "common_mistakes": ["Fingers not curled", "Wrong thumb position", "Too loose"]},
    "F": {"description": "Index + thumb circle, other fingers up",
          "tips": ["Touch index tip to thumb", "Other 3 fingers straight", "Clear circle"],
          "common_mistakes": ["Circle not closed", "Other fingers bent", "Index position wrong"]},
    "G": {"description": "Index and thumb pointing sideways",
          "tips": ["Point index sideways", "Thumb parallel to index", "Other fingers closed"],
          "common_mistakes": ["Fingers not parallel", "Wrong thumb", "Other fingers open"]},
    "H": {"description": "Index and middle together, pointing sideways",
          "tips": ["Extend index and middle together", "Point sideways", "Keep touching"],
          "common_mistakes": ["Fingers apart", "Not sideways", "Other fingers open"]},
    "I": {"description": "Pinky up, fist closed",
          "tips": ["Only pinky extended", "Keep it straight", "Close all others"],
          "common_mistakes": ["Pinky bent", "Others not closed", "Thumb sticking out"]},
    "J": {"description": "Pinky up + draw J in air (motion sign)",
          "tips": ["Start like I", "Draw J downward", "Smooth arc"],
          "common_mistakes": ["No motion", "Wrong direction", "Pinky bent"]},
    "K": {"description": "V shape + thumb touches middle finger",
          "tips": ["Index and middle in V", "Thumb touches middle", "Others closed"],
          "common_mistakes": ["Thumb not touching", "V unclear", "Fingers bent"]},
    "L": {"description": "L with index up and thumb out",
          "tips": ["Index straight up", "Thumb straight out", "90-degree angle"],
          "common_mistakes": ["Not 90 degrees", "Bent fingers", "Others not closed"]},
    "M": {"description": "Three fingers draped over thumb",
          "tips": ["Index, middle, ring over thumb", "Pinky tucked", "Clear shape"],
          "common_mistakes": ["Wrong finger count", "Thumb unclear", "Not draped"]},
    "N": {"description": "Two fingers over thumb",
          "tips": ["Index and middle over thumb", "Others closed", "Similar to M but 2 fingers"],
          "common_mistakes": ["Three fingers (that's M)", "Thumb not visible", "Bent"]},
    "O": {"description": "All fingertips touch thumb",
          "tips": ["All tips to thumb", "Round circle", "Clear O shape"],
          "common_mistakes": ["Circle not closed", "Not all touching", "Not round"]},
    "P": {"description": "K shape pointing downward",
          "tips": ["Form K", "Point down at angle", "Thumb touches middle"],
          "common_mistakes": ["Wrong angle", "Looks like K", "Thumb wrong"]},
    "Q": {"description": "G shape pointing downward",
          "tips": ["Form G", "Point downward", "Thumb and index clear"],
          "common_mistakes": ["Pointing sideways (G)", "Fingers unclear", "Wrong angle"]},
    "R": {"description": "Index and middle crossed",
          "tips": ["Cross index over middle", "Fingers straight", "Others closed"],
          "common_mistakes": ["Not crossed", "Bent fingers", "Others open"]},
    "S": {"description": "Fist with thumb across fingers",
          "tips": ["Tight fist", "Thumb across front", "Cover all fingers"],
          "common_mistakes": ["Thumb on side", "Fist loose", "Thumb not covering"]},
    "T": {"description": "Thumb between index and middle",
          "tips": ["Insert thumb between index and middle", "Make fist", "Thumb pokes through"],
          "common_mistakes": ["Thumb not visible", "Wrong position", "Not closed"]},
    "U": {"description": "Index and middle together pointing up",
          "tips": ["Both up and touching", "Straight up", "Not sideways (H)"],
          "common_mistakes": ["Fingers apart (V)", "Sideways (H)", "Bent"]},
    "V": {"description": "V shape — index and middle apart",
          "tips": ["Spread index and middle", "Clear V", "Straight fingers"],
          "common_mistakes": ["Together (U)", "Bent", "V too wide/narrow"]},
    "W": {"description": "Three fingers up spread in W",
          "tips": ["Index, middle, ring up", "Spread apart", "Keep straight"],
          "common_mistakes": ["Fingers together", "Pinky up (4)", "Bent"]},
    "X": {"description": "Index finger hooked like X",
          "tips": ["Bend index at first knuckle", "Hook shape", "Others closed"],
          "common_mistakes": ["Too straight", "Wrong joint", "Others open"]},
    "Y": {"description": "Thumb and pinky out (shaka)",
          "tips": ["Extend thumb and pinky", "Spread them", "Close others"],
          "common_mistakes": ["Others not closed", "Thumb/pinky bent", "Not spread"]},
    "Z": {"description": "Draw Z in air with index finger (motion sign)",
          "tips": ["Start top-left", "Draw Z shape", "Index extended"],
          "common_mistakes": ["No motion", "Wrong shape", "Wrong starting point"]},
}


@app.get("/learn/{letter}")
async def learn(letter: str):
    letter = letter.upper()
    content = LEARNING_CONTENT.get(letter)
    if not content:
        raise HTTPException(status_code=404, detail=f"Letter '{letter}' not found")
    return content


@app.post("/practice/{letter}")
async def practice(
    letter: str,
    file: UploadFile = File(...),
    lang: str = Form("ASL"),
):
    target = letter.upper()
    lang   = lang.upper()
    try:
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents))
        feat     = extract_landmarks_from_pil(image)

        if feat is None:
            return {
                "correct": False, "target": target, "your_sign": "?",
                "confidence": 0.0, "top_3": [],
                "feedback_message": "No hand detected — make sure your hand is visible",
                "hand_detected": False,
            }

        result = registry.predict(lang, feat)
        pred   = result["prediction"]
        conf   = result["confidence"]

        is_correct = pred == target
        if is_correct:
            msg = ("Perfect! Excellent form!" if conf > 0.9
                   else "Good job! Try to be more precise." if conf > 0.7
                   else "Correct, but could be clearer.")
        else:
            msg = f"That looks like {pred}, not {target}. Check the tips and try again!"

        return {
            "correct": is_correct, "target": target, "your_sign": pred,
            "confidence": conf, "top_3": result["top_3"],
            "feedback_message": msg, "hand_detected": True,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# ElevenLabs Text-to-Speech
# ─────────────────────────────────────────────────────────────────────────────

# Voice IDs for different output languages / accents
# Users can override via the voice_id form field
ELEVENLABS_VOICES = {
    "en":    "EXAVITQu4vr4xnSDxMaL",   # Sarah  (clear American English)
    "en-gb": "onwK4e9ZLuTAKqWW03F9",   # Daniel (British English)
    "hi":    "EXAVITQu4vr4xnSDxMaL",   # fallback — ElevenLabs Hindi support TBD
    "fr":    "onwK4e9ZLuTAKqWW03F9",   # fallback
    "default": "EXAVITQu4vr4xnSDxMaL",
}


def _get_eleven_client(request_key: str = ""):
    """
    Return ElevenLabs client.
    Priority: header key  >  ELEVENLABS_API_KEY env var.
    """
    api_key = request_key or os.environ.get("ELEVENLABS_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="ElevenLabs API key not set. Click the ElevenLabs button in the UI to add your key."
        )
    try:
        from elevenlabs.client import ElevenLabs
        return ElevenLabs(api_key=api_key)
    except ImportError:
        raise HTTPException(status_code=503,
                            detail="elevenlabs package not installed: pip install elevenlabs")


@app.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    voice_id: str = Form(""),
    lang: str = Form("en"),
    model_id: str = Form("eleven_multilingual_v2"),
    x_elevenlabs_key: str = Header(default="", alias="X-ElevenLabs-Key"),
):
    """
    Convert text to speech using ElevenLabs.

    Returns MP3 audio stream.

    Parameters:
      text     – text to speak
      voice_id – ElevenLabs voice ID (optional, auto-selected by lang)
      lang     – language hint: en | en-gb | hi | fr  (default: en)
      model_id – ElevenLabs model (default: eleven_multilingual_v2)
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="text is empty")

    vid = voice_id or ELEVENLABS_VOICES.get(lang, ELEVENLABS_VOICES["default"])
    client = _get_eleven_client(x_elevenlabs_key)

    try:
        audio_gen = client.text_to_speech.convert(
            voice_id=vid,
            text=text,
            model_id=model_id,
            output_format="mp3_44100_128",
        )
        # Collect generator into bytes
        audio_bytes = b"".join(audio_gen)
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={"Content-Disposition": 'inline; filename="speech.mp3"'},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ElevenLabs error: {e}")


@app.get("/tts/voices")
async def list_voices(x_elevenlabs_key: str = Header(default="", alias="X-ElevenLabs-Key")):
    """List available ElevenLabs voices."""
    client = _get_eleven_client(x_elevenlabs_key)
    try:
        result = client.voices.get_all()
        voices = [{"id": v.voice_id, "name": v.name,
                   "labels": v.labels or {}}
                  for v in result.voices]
        return {"voices": voices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tts/status")
async def tts_status():
    """Check whether ElevenLabs is configured."""
    has_key = bool(os.environ.get("ELEVENLABS_API_KEY", ""))
    return {"elevenlabs_configured": has_key,
            "hint": "Set ELEVENLABS_API_KEY environment variable to enable Sign-to-Speech"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
