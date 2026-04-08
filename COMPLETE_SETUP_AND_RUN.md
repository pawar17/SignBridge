# 🚀 Complete Setup & Run Guide - SignBridge MVP

## ⚡ Quick Version (Copy-Paste These Commands)

```bash
# 1. Install PyTorch (CPU version for speed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Install other dependencies
pip install fastapi uvicorn pandas numpy tqdm pillow python-multipart

# 3. Train model (15-20 minutes)
python scripts/training/train_simple_model.py --epochs 5 --batch-size 64

# 4. Start backend (in new terminal)
python backend/api/main.py

# 5. Open frontend in browser
# Double-click: frontend/index.html
```

---

## 📋 Detailed Step-by-Step

### Step 1: Install Dependencies (5 minutes)

**Install PyTorch (CPU version - faster to install):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Or if you have CUDA GPU:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Install FastAPI and other dependencies:**
```bash
pip install fastapi uvicorn[standard] pandas numpy tqdm pillow python-multipart
```

**Verify installation:**
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import fastapi; print('FastAPI installed!')"
```

---

### Step 2: Train the Model (15-20 minutes)

```bash
python scripts/training/train_simple_model.py --epochs 5 --batch-size 64
```

**What happens:**
- Loads 27,456 training images from ASL MNIST
- Trains simple CNN for 5 epochs (~3-4 min each)
- Saves best model to `models/checkpoints/best_model.pth`
- Target accuracy: 65-75% (good enough for MVP!)

**Training output:**
```
Loaded 27,456 samples from training CSV
Loaded 7,173 samples from test CSV
Using device: cpu

Epoch 1/5
Training: 100%|████████| 429/429 [03:15<00:00]
Validation: 100%|████████| 112/112 [00:45<00:00]
Train Loss: 0.5234 | Train Acc: 85.23%
Val Loss: 0.7123 | Val Acc: 78.45%
✓ Saved best model

...

Training Complete!
Best Validation Accuracy: 78.45%
Models saved to: models/checkpoints
```

**Optional: Train longer for better accuracy:**
```bash
python scripts/training/train_simple_model.py --epochs 20
```

---

### Step 3: Start Backend API

**Open Terminal 1:**
```bash
cd C:\Users\aadya\Coding_Projects\SignBridge
python backend/api/main.py
```

**Expected output:**
```
Loading model...
Using device: cpu
Model loaded successfully! Validation accuracy: 78.45%
Classes: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
INFO:     Uvicorn running on http://0.0.0.0:8000
```

✅ **API is now running!** Keep this terminal open.

---

### Step 4: Test API (Optional)

**Open Terminal 2:**
```bash
python test_api.py
```

**Or test in browser:**
- API Info: http://localhost:8000
- Health Check: http://localhost:8000/health
- API Docs: http://localhost:8000/docs (Interactive Swagger UI!)

---

### Step 5: Open Frontend

**Method 1: Double-click**
- Navigate to `SignBridge/frontend/`
- Double-click `index.html`

**Method 2: Full path in browser**
```
file:///C:/Users/aadya/Coding_Projects/SignBridge/frontend/index.html
```

**Method 3: Simple HTTP server (optional)**
```bash
cd frontend
python -m http.server 3000
# Then open: http://localhost:3000
```

---

### Step 6: Use the App! 🎉

1. **Allow camera access** when prompted
2. **Click "Start Camera"**
3. **Show an ASL sign** (A, B, C, etc.)
4. **Click "Predict Sign"** - Get instant prediction!
5. **Or enable "Auto Mode"** - Predictions every 2 seconds

---

## 🎯 Quick Demo Flow

```
1. Show 'A' → Predict → Should say "A"
2. Show 'B' → Predict → Should say "B"
3. Show 'C' → Predict → Should say "C"
4. Enable Auto Mode → Show different signs continuously
```

---

## 📊 What You Get

### Frontend Features:
- ✅ Live webcam feed
- ✅ Real-time predictions
- ✅ Confidence scores (0-100%)
- ✅ Top 3 predictions
- ✅ Manual or Auto mode
- ✅ Supported alphabet reference

### Backend Features:
- ✅ REST API with FastAPI
- ✅ Model inference
- ✅ CORS enabled
- ✅ Error handling
- ✅ Interactive docs (Swagger)

---

## ❓ Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### "Model checkpoint not found"
```bash
# Train the model first!
python scripts/training/train_simple_model.py --epochs 5
```

### "Cannot connect to API"
```bash
# Make sure backend is running:
python backend/api/main.py
# Should show "Uvicorn running on http://0.0.0.0:8000"
```

### "Camera access denied"
- Allow camera permissions in browser
- Try Chrome or Edge (best compatibility)
- Check if another app is using camera

### Backend crashes with "Address already in use"
```bash
# Kill process on port 8000:
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -i :8000
kill -9 <PID>
```

---

## 🎨 Tips for Best Demo

1. **Good Lighting** - Well-lit room, light on your hand
2. **Plain Background** - Reduces noise
3. **Center Hand** - Keep hand in frame center
4. **Hold Steady** - 1-2 seconds per sign
5. **Close to Camera** - But not too close

**Reference ASL Alphabet:**
- Google "ASL finger spelling alphabet"
- Practice a few letters before demo
- Focus on clear, distinct signs

---

## 📈 Expected Performance

| Metric | Value |
|--------|-------|
| **Training Time** | 15-20 min (5 epochs) |
| **Model Size** | ~2-3 MB |
| **Inference Time** | 50-100ms per image |
| **Accuracy** | 65-80% (MVP baseline) |
| **Supported Signs** | 24 letters (A-Y, no J/Z) |

**Note:** J and Z require motion, not included in static ASL MNIST dataset.

---

## 🚀 Next Steps After Demo

### Immediate Improvements:
1. **Train Longer** - 20-50 epochs for 85-90% accuracy
2. **Better Preprocessing** - Hand cropping, better normalization
3. **Data Augmentation** - Rotation, scaling, brightness
4. **More Data** - Add custom ASL dataset

### Future Features:
1. **Video Recognition** - Not just static images
2. **Word Recognition** - Multiple signs → words
3. **Text-to-Sign** - Avatar generation
4. **Multiple Languages** - ISL, BSL, etc.
5. **Mobile App** - React Native version

---

## 📝 Document Your Work

**Update PROJECT_REFERENCE.md:**
```markdown
## Sprint Results (January 20, 2024)

### Completed:
- ✅ Trained CNN model: 78% accuracy on ASL MNIST
- ✅ Built FastAPI backend with /predict endpoint
- ✅ Created web UI with webcam integration
- ✅ End-to-end working demo

### Performance:
- Training: 15 minutes (5 epochs)
- Inference: ~80ms per prediction
- Dataset: 27,456 training, 7,173 test images

### Demo Video:
[Link to demo video/screenshots]
```

---

## 🎉 You're Done!

You've built a complete ASL recognition system:
- ✅ Deep learning model
- ✅ REST API backend
- ✅ Web interface
- ✅ Real-time predictions

**Time to demo: ~2 hours from start to finish!**

Now share it, get feedback, and iterate! 🚀

---

## 📞 Next Session Ideas

1. Deploy to cloud (Heroku, AWS, etc.)
2. Add more sign languages
3. Mobile version
4. Sentence recognition
5. Learning mode for users
6. Social features (share signs)

**Keep building! 💪**
