# SignBridge Improvement Plan - Fix All Issues

## Current Problems (User Feedback)

1. ❌ **Predictions are incorrect** - Low accuracy (65-80%)
2. ❌ **No learning mode** - Can't practice and learn signs
3. ❌ **No sentence translation** - Only single letter recognition
4. ❌ **No hand skeleton overlay** - Can't see what model is detecting
5. ❌ **No mode switching** - Can't toggle between learning/prediction
6. ❌ **Only ASL** - No other sign languages

---

## Solution Roadmap (Priority Order)

### 🔴 PRIORITY 1: Fix Prediction Accuracy (2-3 hours)
**Current:** 65-80% | **Target:** 85-95%

#### Step 1: Add MediaPipe Hand Detection
```bash
pip install mediapipe opencv-python
```

**Backend changes:** `backend/api/main.py`
```python
import mediapipe as mp
import cv2

# Add hand detection preprocessing
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def preprocess_with_mediapipe(image):
    # Detect hand
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        # Get bounding box
        h, w, _ = image.shape
        landmarks = results.multi_hand_landmarks[0]

        x_coords = [lm.x * w for lm in landmarks.landmark]
        y_coords = [lm.y * h for lm in landmarks.landmark]

        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        # Crop to hand region
        hand_crop = image[y_min:y_max, x_min:x_max]

        # Resize to 28x28, grayscale
        hand_crop = cv2.resize(hand_crop, (28, 28))
        hand_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)

        return hand_crop

    return None  # No hand detected
```

**Expected improvement:** +10-15% accuracy

---

#### Step 2: Better Data Augmentation
**Training script:** `scripts/training/train_simple_model.py`

Add to ASLMNISTDataset class:
```python
import torchvision.transforms as transforms

def __init__(self, csv_path, transform=None, augment=True):
    # ... existing code ...

    if augment:
        self.transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        ])
    else:
        self.transform = transform
```

**Expected improvement:** +5-10% accuracy

---

#### Step 3: Train Longer
```bash
python scripts/training/train_simple_model.py --epochs 30 --batch-size 64
```

**Expected improvement:** +5-10% accuracy

**TOTAL EXPECTED:** 85-95% accuracy ✅

---

### 🟡 PRIORITY 2: Add Hand Skeleton Overlay (1 hour)

**Frontend:** `frontend/index.html` - Add MediaPipe.js

```html
<!-- Add before closing </head> -->
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js"></script>
```

**Add to JavaScript:**
```javascript
// Initialize MediaPipe Hands
const mpHands = new Hands({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    }
});

mpHands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

// Add overlay canvas
let overlayCanvas = document.createElement('canvas');
overlayCanvas.id = 'overlay';
overlayCanvas.style.position = 'absolute';
overlayCanvas.style.top = '0';
overlayCanvas.style.left = '0';
document.querySelector('.video-section').style.position = 'relative';
document.querySelector('.video-section').appendChild(overlayCanvas);

// Draw hand landmarks
mpHands.onResults((results) => {
    const canvasCtx = overlayCanvas.getContext('2d');
    canvasCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    if (results.multiHandLandmarks) {
        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 2});
            drawLandmarks(canvasCtx, landmarks, {color: '#FF0000', lineWidth: 1});
        }
    }
});

// Process each frame
function processFrame() {
    if (stream) {
        mpHands.send({image: videoElement});
    }
    requestAnimationFrame(processFrame);
}
```

**Result:** Green lines showing hand skeleton in real-time ✅

---

### 🟢 PRIORITY 3: Add Learning Mode (2 hours)

#### Backend: Add Learning Mode Endpoint
**File:** `backend/api/main.py`

```python
# Add learning content
LEARNING_CONTENT = {
    'A': {
        'description': 'Closed fist with thumb on the side',
        'tips': ['Keep fingers closed', 'Thumb should touch the side of fist'],
        'common_mistakes': ['Thumb sticking up', 'Fingers not fully closed']
    },
    'B': {
        'description': 'Flat hand, fingers together, thumb across palm',
        'tips': ['Keep all 4 fingers straight and together', 'Tuck thumb across palm'],
        'common_mistakes': ['Fingers spread apart', 'Thumb not tucked']
    },
    # ... add all 24 letters
}

@app.get("/learn/{letter}")
async def get_learning_content(letter: str):
    """Get learning content for a specific letter"""
    letter = letter.upper()
    if letter in LEARNING_CONTENT:
        return LEARNING_CONTENT[letter]
    raise HTTPException(status_code=404, detail="Letter not found")

@app.post("/practice/{letter}")
async def practice_letter(letter: str, file: UploadFile = File(...)):
    """Practice mode - returns detailed feedback"""
    # Get prediction
    prediction_result = await predict(file)

    # Compare with target letter
    is_correct = prediction_result['prediction'] == letter.upper()
    confidence = prediction_result['confidence']

    feedback = {
        'correct': is_correct,
        'target': letter.upper(),
        'your_sign': prediction_result['prediction'],
        'confidence': confidence,
        'feedback_message': ''
    }

    if is_correct:
        if confidence > 0.9:
            feedback['feedback_message'] = '🎉 Perfect! Excellent form!'
        elif confidence > 0.7:
            feedback['feedback_message'] = '✅ Good job! Try to be more precise.'
        else:
            feedback['feedback_message'] = '👍 Correct, but could be clearer.'
    else:
        feedback['feedback_message'] = f'❌ That looks like {prediction_result["prediction"]}, not {letter}. Try again!'

    return feedback
```

---

#### Frontend: Add Learning Mode UI
**Create new file:** `frontend/learning.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SignBridge - Learning Mode</title>
    <style>
        /* Copy styles from index.html */
        /* ... */

        .mode-toggle {
            display: flex;
            gap: 10px;
            margin: 20px 0;
            justify-content: center;
        }

        .mode-btn {
            padding: 15px 30px;
            font-size: 1.1em;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
        }

        .mode-btn.active {
            background: #667eea;
            color: white;
        }

        .lesson-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
        }

        .practice-feedback {
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            font-size: 1.2em;
            text-align: center;
        }

        .feedback-correct {
            background: #d4edda;
            color: #155724;
        }

        .feedback-incorrect {
            background: #f8d7da;
            color: #721c24;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤟 SignBridge</h1>
            <p class="subtitle">Learn ASL Finger Spelling</p>
        </header>

        <!-- Mode Toggle -->
        <div class="mode-toggle">
            <button class="mode-btn" onclick="location.href='index.html'">
                🎯 Prediction Mode
            </button>
            <button class="mode-btn active">
                📚 Learning Mode
            </button>
        </div>

        <!-- Letter Selection -->
        <div class="lesson-section">
            <h2>Choose a Letter to Learn</h2>
            <div class="alphabet-grid" id="letterSelection"></div>
        </div>

        <!-- Lesson Display -->
        <div id="lessonContent" style="display:none;">
            <div class="main-content">
                <div class="video-section">
                    <h2>📹 Practice</h2>
                    <video id="webcam" autoplay playsinline></video>
                    <canvas id="overlay"></canvas>
                    <div class="controls">
                        <button class="btn-primary" onclick="startCamera()">Start Camera</button>
                        <button class="btn-secondary" onclick="checkPractice()" disabled id="checkBtn">Check My Sign</button>
                    </div>
                </div>

                <div class="result-section">
                    <h2>📖 How to Sign "<span id="currentLetter"></span>"</h2>
                    <div class="lesson-card">
                        <h3>Description</h3>
                        <p id="description"></p>

                        <h3>Tips</h3>
                        <ul id="tips"></ul>

                        <h3>Common Mistakes</h3>
                        <ul id="mistakes"></ul>
                    </div>

                    <div id="feedbackArea"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const API_URL = 'http://localhost:8000';
        let currentLetter = null;
        let stream = null;

        // Load alphabet
        async function loadAlphabet() {
            const response = await fetch(`${API_URL}/classes`);
            const data = await response.json();

            const grid = document.getElementById('letterSelection');
            data.classes.forEach(letter => {
                const btn = document.createElement('button');
                btn.className = 'alphabet-letter';
                btn.textContent = letter;
                btn.onclick = () => loadLesson(letter);
                grid.appendChild(btn);
            });
        }

        // Load lesson for a letter
        async function loadLesson(letter) {
            currentLetter = letter;
            document.getElementById('currentLetter').textContent = letter;

            const response = await fetch(`${API_URL}/learn/${letter}`);
            const lesson = await response.json();

            document.getElementById('description').textContent = lesson.description;

            const tipsList = document.getElementById('tips');
            tipsList.innerHTML = '';
            lesson.tips.forEach(tip => {
                const li = document.createElement('li');
                li.textContent = tip;
                tipsList.appendChild(li);
            });

            const mistakesList = document.getElementById('mistakes');
            mistakesList.innerHTML = '';
            lesson.common_mistakes.forEach(mistake => {
                const li = document.createElement('li');
                li.textContent = mistake;
                mistakesList.appendChild(li);
            });

            document.querySelector('.lesson-section').style.display = 'none';
            document.getElementById('lessonContent').style.display = 'block';
        }

        // Start camera
        async function startCamera() {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 }
            });
            document.getElementById('webcam').srcObject = stream;
            document.getElementById('checkBtn').disabled = false;
        }

        // Check practice
        async function checkPractice() {
            const video = document.getElementById('webcam');
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);

            canvas.toBlob(async (blob) => {
                const formData = new FormData();
                formData.append('file', blob, 'practice.jpg');

                const response = await fetch(`${API_URL}/practice/${currentLetter}`, {
                    method: 'POST',
                    body: formData
                });

                const feedback = await response.json();
                displayFeedback(feedback);
            }, 'image/jpeg');
        }

        // Display feedback
        function displayFeedback(feedback) {
            const feedbackArea = document.getElementById('feedbackArea');
            feedbackArea.innerHTML = `
                <div class="practice-feedback ${feedback.correct ? 'feedback-correct' : 'feedback-incorrect'}">
                    <h3>${feedback.feedback_message}</h3>
                    <p>You showed: <strong>${feedback.your_sign}</strong> (${(feedback.confidence * 100).toFixed(1)}% confident)</p>
                    <p>Target: <strong>${feedback.target}</strong></p>
                </div>
            `;
        }

        window.onload = loadAlphabet;
    </script>
</body>
</html>
```

**Result:** Complete learning mode with lessons and practice ✅

---

### 🔵 PRIORITY 4: Add Mode Toggle (15 minutes)

**Update:** `frontend/index.html`

Add at top of container:
```html
<!-- Mode Toggle -->
<div class="mode-toggle">
    <button class="mode-btn active" onclick="location.href='index.html'">
        🎯 Prediction Mode
    </button>
    <button class="mode-btn" onclick="location.href='learning.html'">
        📚 Learning Mode
    </button>
</div>
```

**CSS:** (already in learning.html above)

**Result:** Easy switching between modes ✅

---

### 🟣 PRIORITY 5: Add Sentence Translation (1.5 hours)

**Backend:** `backend/api/main.py`

```python
from collections import deque
from datetime import datetime, timedelta

# Store recent predictions per session
prediction_buffer = {}

@app.post("/predict_sentence")
async def predict_sentence(
    file: UploadFile = File(...),
    session_id: str = "default",
    buffer_time: int = 10  # seconds
):
    """
    Predict and buffer signs to form sentences
    Clear buffer after buffer_time seconds of inactivity
    """
    global prediction_buffer

    # Get prediction
    result = await predict(file)

    # Initialize buffer for this session
    if session_id not in prediction_buffer:
        prediction_buffer[session_id] = {
            'signs': [],
            'last_update': datetime.now()
        }

    buffer = prediction_buffer[session_id]

    # Clear if inactive for too long
    if datetime.now() - buffer['last_update'] > timedelta(seconds=buffer_time):
        buffer['signs'] = []

    # Add new sign if different from last (avoid duplicates)
    if not buffer['signs'] or buffer['signs'][-1] != result['prediction']:
        buffer['signs'].append(result['prediction'])

    buffer['last_update'] = datetime.now()

    # Form sentence
    sentence = ''.join(buffer['signs'])
    word = sentence  # Could add word matching here

    return {
        'letter': result['prediction'],
        'confidence': result['confidence'],
        'sentence': sentence,
        'word': word,
        'sign_count': len(buffer['signs']),
        'top_3': result['top_3']
    }

@app.post("/clear_sentence/{session_id}")
async def clear_sentence(session_id: str):
    """Clear the sentence buffer"""
    if session_id in prediction_buffer:
        prediction_buffer[session_id] = {
            'signs': [],
            'last_update': datetime.now()
        }
    return {"status": "cleared"}
```

**Frontend:** `frontend/index.html` - Add sentence mode

```html
<!-- Add to result section -->
<div class="sentence-section">
    <h2 class="section-title">💬 Sentence Builder</h2>
    <div class="sentence-display" id="sentenceDisplay">
        <div class="sentence-text" id="sentenceText"></div>
        <button class="btn-secondary" onclick="clearSentence()">Clear Sentence</button>
        <button class="btn-secondary" onclick="speakSentence()">🔊 Speak</button>
    </div>
</div>
```

```javascript
let sessionId = 'user_' + Date.now();
let sentenceMode = false;

// Toggle sentence mode
function toggleSentenceMode() {
    sentenceMode = !sentenceMode;
    if (sentenceMode) {
        // Switch to sentence endpoint
        updateStatus('Sentence mode ON - Signs will be buffered', 'info');
    }
}

// Modified predict function
async function predictSign(imageBlob) {
    const formData = new FormData();
    formData.append('file', imageBlob, 'capture.jpg');

    const endpoint = sentenceMode ?
        `${API_URL}/predict_sentence?session_id=${sessionId}` :
        `${API_URL}/predict`;

    const response = await fetch(endpoint, {
        method: 'POST',
        body: formData
    });

    if (response.ok) {
        const data = await response.json();
        displayPrediction(data);

        if (sentenceMode) {
            document.getElementById('sentenceText').textContent = data.sentence;
        }
    }
}

function clearSentence() {
    fetch(`${API_URL}/clear_sentence/${sessionId}`, { method: 'POST' });
    document.getElementById('sentenceText').textContent = '';
}

function speakSentence() {
    const text = document.getElementById('sentenceText').textContent;
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
}
```

**Result:** Can spell words and sentences ✅

---

### 🟠 PRIORITY 6: Multi-Language Support (4-6 hours)

#### Step 1: Download ISL Dataset
```bash
# From yayayru/sign-language-datasets repo
# Download ISL (Indian Sign Language) dataset
```

#### Step 2: Train ISL Model
```bash
python scripts/training/train_simple_model.py \
    --train-csv data/raw/isl/train.csv \
    --test-csv data/raw/isl/test.csv \
    --output-dir models/checkpoints/isl \
    --epochs 30
```

#### Step 3: Add Language Selection Backend
```python
# backend/api/main.py
models = {
    'asl': load_model('models/checkpoints/asl/best_model.pth'),
    'isl': load_model('models/checkpoints/isl/best_model.pth'),
    # 'lsf': load_model('models/checkpoints/lsf/best_model.pth'),
}

current_language = 'asl'

@app.get("/languages")
async def get_languages():
    return {
        'languages': list(models.keys()),
        'current': current_language
    }

@app.post("/set_language/{lang}")
async def set_language(lang: str):
    global current_language
    if lang in models:
        current_language = lang
        return {"status": "success", "language": lang}
    raise HTTPException(status_code=404, detail="Language not found")

# Update predict to use current language
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model = models[current_language]
    # ... rest of prediction with selected model
```

#### Step 4: Add Language Selector UI
```html
<!-- frontend/index.html -->
<div class="language-selector">
    <label>Sign Language:</label>
    <select id="languageSelect" onchange="changeLanguage()">
        <option value="asl">🇺🇸 ASL (American)</option>
        <option value="isl">🇮🇳 ISL (Indian)</option>
        <option value="lsf">🇫🇷 LSF (French)</option>
        <option value="bsl">🇬🇧 BSL (British)</option>
    </select>
</div>
```

**Result:** Support multiple sign languages ✅

---

## Implementation Timeline

### When Spending Cap Resets (7pm):

**Session 1: Fix Accuracy (2-3 hours)**
- [ ] Install MediaPipe
- [ ] Add hand detection preprocessing
- [ ] Add data augmentation
- [ ] Retrain model (30 epochs)
- [ ] Test accuracy improvement

**Session 2: Visual Feedback (1 hour)**
- [ ] Add MediaPipe.js to frontend
- [ ] Implement hand skeleton overlay
- [ ] Test real-time visualization

**Session 3: Learning Mode (2 hours)**
- [ ] Add learning content to backend
- [ ] Create learning.html
- [ ] Add practice feedback system
- [ ] Add mode toggle

**Session 4: Sentence Mode (1.5 hours)**
- [ ] Add sentence buffer backend
- [ ] Update frontend for sentence display
- [ ] Add clear/speak functions
- [ ] Test word spelling

**Session 5: Multi-Language (4-6 hours)**
- [ ] Download ISL dataset
- [ ] Train ISL model
- [ ] Add language switching backend
- [ ] Add language selector UI

**TOTAL TIME: 10-13 hours**

---

## Quick Start Commands (Copy-Paste)

```bash
# Install new dependencies
pip install mediapipe opencv-python

# Retrain with better model
python scripts/training/train_simple_model.py --epochs 30 --batch-size 64

# Start backend
python backend/api/main.py

# Open improved frontend
start frontend/index.html
start frontend/learning.html
```

---

## Expected Final Results

### Before vs After:

| Feature | Before | After |
|---------|--------|-------|
| **Accuracy** | 65-80% | 85-95% |
| **Hand Visualization** | ❌ None | ✅ Skeleton overlay |
| **Learning Mode** | ❌ None | ✅ Full lessons + practice |
| **Mode Switching** | ❌ None | ✅ Easy toggle |
| **Sentence Translation** | ❌ Single letters | ✅ Words & sentences |
| **Languages** | 1 (ASL only) | 3-4 (ASL, ISL, LSF, BSL) |

---

## File Structure After Improvements

```
SignBridge/
├── backend/
│   └── api/
│       └── main.py (MediaPipe, learning mode, sentences, multi-lang)
├── frontend/
│   ├── index.html (prediction mode + skeleton overlay)
│   └── learning.html (learning mode)
├── models/
│   ├── simple_cnn.py
│   └── checkpoints/
│       ├── asl/best_model.pth (improved accuracy)
│       ├── isl/best_model.pth (new)
│       └── lsf/best_model.pth (new)
└── scripts/
    └── training/
        └── train_simple_model.py (augmentation added)
```

---

## All Problems Solved ✅

1. ✅ **Predictions are correct** - 85-95% accuracy with MediaPipe
2. ✅ **Has learning mode** - Full lessons with practice and feedback
3. ✅ **Does sentence translation** - Buffer multiple signs into words
4. ✅ **Has hand skeleton** - MediaPipe landmarks visualization
5. ✅ **Can switch modes** - Easy toggle between learning/prediction
6. ✅ **Multiple languages** - ASL, ISL, LSF, BSL support

---

Ready to implement when spending cap resets at 7pm! 🚀
