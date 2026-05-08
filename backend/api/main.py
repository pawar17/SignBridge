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
            min_hand_detection_confidence=0.3,
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

# per-session sentence buffer
prediction_buffer: dict = defaultdict(lambda: {"signs": [], "last_update": datetime.now()})

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
def predict(
    file: UploadFile = File(...),
    lang: str = Form("ASL"),
):
    """
    Predict sign letter from a webcam frame.

    Parameters:
      file  – JPEG/PNG image (form-data)
      lang  – ASL | BSL | ISL  (default ASL)
    """
    lang = lang.upper()

    if not registry.models and not registry.label_maps:
        raise HTTPException(
            status_code=503,
            detail="No models loaded. Run: python scripts/training/extract_and_train_landmarks.py"
        )

    try:
        contents = file.file.read()
        image    = Image.open(io.BytesIO(contents))
        feat     = extract_landmarks_from_pil(image)

        if feat is None:
            # no hand detected — return low-confidence placeholder
            return PredictionResponse(
                prediction="?",
                confidence=0.0,
                top_3=[],
                success=False,
                hand_detected=False,
                lang=lang,
            )

        result = registry.predict(lang, feat)
        return PredictionResponse(
            **result,
            success=True,
            hand_detected=True,
            lang=lang,
        )

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_sentence")
def predict_sentence(
    file: UploadFile = File(...),
    lang: str = Form("ASL"),
    session_id: str = Form("default"),
    buffer_time: int = Form(10),
):
    """Predict + build up a sentence letter-by-letter."""
    lang = lang.upper()

    try:
        contents = file.file.read()
        image    = Image.open(io.BytesIO(contents))
        feat     = extract_landmarks_from_pil(image)

        if feat is None:
            return {
                "letter": "?",
                "confidence": 0.0,
                "sentence": "".join(prediction_buffer[session_id]["signs"]),
                "word": "".join(prediction_buffer[session_id]["signs"]),
                "sign_count": len(prediction_buffer[session_id]["signs"]),
                "top_3": [],
                "success": False,
                "hand_detected": False,
                "lang": lang,
            }

        result = registry.predict(lang, feat)
        letter = result["prediction"]

        buf = prediction_buffer[session_id]
        if datetime.now() - buf["last_update"] > timedelta(seconds=buffer_time):
            buf["signs"] = []
        if not buf["signs"] or buf["signs"][-1] != letter:
            buf["signs"].append(letter)
        buf["last_update"] = datetime.now()

        sentence = "".join(buf["signs"])
        return {
            "letter":       letter,
            "confidence":   result["confidence"],
            "sentence":     sentence,
            "word":         sentence,
            "sign_count":   len(buf["signs"]),
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
    prediction_buffer[session_id] = {"signs": [], "last_update": datetime.now()}
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
def practice(
    letter: str,
    file: UploadFile = File(...),
    lang: str = Form("ASL"),
):
    target = letter.upper()
    lang   = lang.upper()
    try:
        contents = file.file.read()
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
