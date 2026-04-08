# 🚀 START HERE - Tonight's Development Sprint

**Time to complete**: 8-12 hours
**Current Sprint**: 1 of 4

---

## Step 1: Quick Setup (5 minutes)

### Windows:
```cmd
quick_setup.bat
```

### Linux/Mac:
```bash
chmod +x quick_setup.sh
./quick_setup.sh
```

**Or manually:**
```bash
cp .env.example .env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pandas numpy tqdm scikit-learn matplotlib
```

---

## Step 2: Train Model (15-20 minutes) ⏱️

```bash
python scripts/training/train_simple_model.py --epochs 5 --batch-size 64
```

**Expected output:**
- Training will process 27,456 samples
- Each epoch takes ~3-4 minutes
- Target accuracy: >60% (baseline MVP)
- Model saved to: `models/checkpoints/best_model.pth`

**While training runs**, proceed to Step 3 to save time!

---

## Step 3: Create Backend (30 minutes)

I'll create the FastAPI backend for you. Copy this command:

```bash
# Backend will be created in next step
# Keep training running in another terminal!
```

---

## Step 4: Create Frontend (45 minutes)

React app with webcam and real-time prediction.

---

## Step 5: Test & Demo (30 minutes)

End-to-end testing and polish.

---

## Quick Status Check

Run this to see what's working:

```bash
# Check if model exists
dir models\checkpoints\best_model.pth   # Windows
ls models/checkpoints/best_model.pth    # Linux/Mac

# Check if training is done
# Look for "Training Complete!" message
```

---

## What We're Building Tonight

```
User shows ASL sign → Webcam captures →
MediaPipe detects hand → Model predicts →
UI shows letter + confidence
```

**Demo will recognize**: A-Z ASL finger spelling (excluding J, Z)

---

## Need Help?

- Check `SPRINT_PLAN.md` for detailed breakdown
- Check `PROJECT_REFERENCE.md` for full documentation
- Training logs in terminal show progress

---

## Let's Go! 🎯

1. ✅ Run setup script
2. ⏳ Start training (it runs in background)
3. ⏳ Build backend (while training)
4. ⏳ Build frontend
5. ⏳ Test and demo!

**Start the setup now, then training!**
