# SignBridge - Tonight's Sprint Plan 🚀

**Goal**: Working MVP with sign recognition by tonight!
**Time Available**: 8-12 hours
**Target**: Real-time ASL finger spelling recognition with web interface

---

## MVP Scope (What We're Building Tonight)

### ✅ In Scope
- ASL finger spelling recognition (A-Z, 0-9)
- Real-time webcam capture
- Simple CNN model (baseline)
- FastAPI backend
- React frontend with basic UI
- End-to-end demo working

### ❌ Out of Scope (Future)
- Text-to-sign generation
- Multiple sign languages (focus on ASL only)
- Complex Transformer models
- Learning mode
- User authentication
- Production deployment

---

## Sprint Breakdown

### 🏃 SPRINT 1: Data & Model (2-3 hours)
**Goal**: Get a working sign recognition model

#### Tasks:
1. **Setup Environment** (15 min)
   - Create .env file
   - Install dependencies
   - Test imports

2. **Process ASL MNIST Data** (30 min)
   - Load CSV data
   - Create simple train/test split
   - Normalize images

3. **Build Simple CNN Model** (45 min)
   - Create models/simple_cnn.py
   - Use PyTorch: Conv layers → FC layers
   - Keep it simple (3-4 conv layers)

4. **Quick Training** (60 min)
   - Train for 5-10 epochs (just to get something working)
   - Target: >60% accuracy (baseline)
   - Save checkpoint

**Deliverable**: Trained model checkpoint that can classify signs

---

### 🏃 SPRINT 2: Backend API (2 hours)
**Goal**: FastAPI server that serves the model

#### Tasks:
1. **Create FastAPI App** (30 min)
   - backend/api/main.py with basic routes
   - Health check endpoint
   - Model loading on startup

2. **Image Processing Endpoint** (45 min)
   - POST /predict endpoint
   - Accept base64 image or file upload
   - Preprocess image
   - Run inference
   - Return prediction + confidence

3. **Add CORS & Error Handling** (15 min)
   - Enable CORS for frontend
   - Proper error responses
   - Request validation

4. **Test API** (30 min)
   - Test with curl/Postman
   - Verify predictions work
   - Check response times

**Deliverable**: Running API on localhost:8000 that classifies ASL signs

---

### 🏃 SPRINT 3: Frontend UI (2-3 hours)
**Goal**: Web interface for real-time recognition

#### Tasks:
1. **Setup React App** (20 min)
   - Create React app
   - Install dependencies (webcam, axios)
   - Basic project structure

2. **Webcam Component** (45 min)
   - Use react-webcam
   - Capture frames every second
   - Display video feed

3. **MediaPipe Integration** (60 min)
   - Add MediaPipe Hands
   - Detect hand in frame
   - Extract hand region
   - Preprocess for model

4. **Prediction Display** (45 min)
   - Send frames to backend
   - Display prediction + confidence
   - Show sign alphabet reference
   - Add basic styling

**Deliverable**: Web UI where user can show ASL signs and get real-time predictions

---

### 🏃 SPRINT 4: Integration & Polish (1-2 hours)
**Goal**: Make it demo-ready

#### Tasks:
1. **End-to-End Testing** (30 min)
   - Test all signs A-Z
   - Check accuracy and latency
   - Fix critical bugs

2. **UI Polish** (30 min)
   - Add loading states
   - Show confidence meter
   - Add instructions
   - Basic CSS styling

3. **Documentation** (30 min)
   - Update PROJECT_REFERENCE.md
   - Add demo instructions
   - Take screenshots/video

**Deliverable**: Demo-ready application!

---

## Technology Choices (Simplified for Speed)

### Model
- **Simple CNN** (not Transformer - too complex for tonight)
- PyTorch with 3-4 conv layers
- Train on ASL MNIST (static images, easier than video)
- Target: 60-70% accuracy (acceptable for demo)

### Backend
- FastAPI (already configured)
- Single predict endpoint
- No database needed for MVP
- No authentication

### Frontend
- React with react-webcam
- MediaPipe Hands (browser-based)
- Axios for API calls
- Basic CSS (no fancy UI library)

---

## Quick Start Commands

### Sprint 1: Data & Model
```bash
# 1. Setup
cp .env.example .env
pip install torch torchvision pandas numpy matplotlib scikit-learn

# 2. Create training script
# We'll create: scripts/train_simple_model.py

# 3. Train
python scripts/train_simple_model.py --epochs 10 --batch-size 64
```

### Sprint 2: Backend
```bash
# 1. Create API
# We'll create: backend/api/main.py (simplified)

# 2. Run server
uvicorn backend.api.main:app --reload --port 8000
```

### Sprint 3: Frontend
```bash
# 1. Create React app
cd frontend
npm create vite@latest . -- --template react
npm install

# 2. Add dependencies
npm install react-webcam axios @mediapipe/hands

# 3. Run
npm run dev
```

---

## Success Criteria

### Minimum (Must Have)
- [ ] Model trains and saves checkpoint
- [ ] API accepts images and returns predictions
- [ ] Frontend shows webcam feed
- [ ] End-to-end: Show sign → Get prediction

### Nice to Have (If Time)
- [ ] Confidence threshold filtering
- [ ] Sign alphabet reference chart
- [ ] Smooth UI with loading states
- [ ] 70%+ accuracy

### Stretch Goals (Bonus)
- [ ] Real-time continuous recognition (not just single frames)
- [ ] Hand detection visualization
- [ ] Top-3 predictions shown
- [ ] Mobile responsive

---

## Time Checkpoints

| Time | Checkpoint | Status |
|------|-----------|--------|
| +1 hour | Environment setup, data loaded | ⏳ |
| +3 hours | Model trained and saved | ⏳ |
| +5 hours | API running and tested | ⏳ |
| +8 hours | Frontend working | ⏳ |
| +10 hours | End-to-end demo working | ⏳ |
| +12 hours | Polished and documented | ⏳ |

---

## Let's Go! 🚀

Starting with Sprint 1: Data & Model

**Current Time**: [Record when starting]
**Target Completion**: Tonight!

Update this file as we progress through each sprint!
