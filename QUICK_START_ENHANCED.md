# SignBridge Enhanced - Quick Start Guide

## 🚀 What's New

The enhanced SignBridge now supports:
- **Spatial-Temporal Models**: Better accuracy with sequence understanding
- **Video Processing**: Full video support (not just single frames)
- **MediaPipe Integration**: Advanced landmark extraction
- **Data Collection Tools**: Record your own sign language data
- **Flexible Architecture**: Supports both simple CNN and advanced models

## 📋 Prerequisites

```bash
# Install additional dependencies
pip install pytorch-lightning albumentations
```

## 🎯 Quick Start Options

### Option 1: Use Existing Simple Model (Current MVP)
Your current setup works! The backend automatically detects and uses the simple CNN model.

### Option 2: Record Custom Data & Train New Model

#### Step 1: Record Signs
```bash
# Record a sign
python scripts/data_collection/sign_recorder.py --sign HELLO --signer-id signer_001

# Record multiple signs
python scripts/data_collection/sign_recorder.py --sign THANK_YOU
python scripts/data_collection/sign_recorder.py --sign PLEASE
```

#### Step 2: Create Annotation File
Create `data/annotations/train_annotations.json`:
```json
[
  {
    "video_path": "custom/HELLO_signer_001_session_001_20240101_120000.mp4",
    "sign_label": "HELLO",
    "gloss": "HELLO",
    "signer_id": "signer_001"
  },
  {
    "video_path": "custom/THANK_YOU_signer_001_session_001_20240101_120100.mp4",
    "sign_label": "THANK_YOU",
    "gloss": "THANK_YOU",
    "signer_id": "signer_001"
  }
]
```

#### Step 3: Preprocess Videos
```bash
# Preprocess training data
python scripts/data_preprocessing/video_preprocessor.py \
  --input-dir data/raw \
  --output-dir data/processed \
  --annotation-file data/annotations/train_annotations.json \
  --split train

# Preprocess validation data
python scripts/data_preprocessing/video_preprocessor.py \
  --input-dir data/raw \
  --output-dir data/processed \
  --annotation-file data/annotations/val_annotations.json \
  --split val
```

#### Step 4: Train Model
```bash
python scripts/training/train_spatial_temporal.py \
  --train-data data/processed/train_processed.pkl \
  --val-data data/processed/val_processed.pkl \
  --batch-size 16 \
  --epochs 50 \
  --output-dir models/checkpoints
```

### Option 3: Use Public Datasets

#### Download WLASL
```python
# See scripts/data_collection/download_wlasl.py (to be created)
# Or manually download from: https://github.com/dxli94/WLASL
```

## 🔧 Current System Status

✅ **Working Now:**
- Simple CNN model (24 letters)
- Basic prediction API
- Learning mode
- Sentence building
- Frontend interface

🔄 **Ready to Use (New Features):**
- Video preprocessing pipeline
- Spatial-temporal model architecture
- Data recording tool
- Enhanced model loader

⏳ **Next Steps:**
1. Collect/process video data
2. Train spatial-temporal model
3. Deploy enhanced model

## 📁 Project Structure

```
signbridge/
├── data/
│   ├── raw/              # Raw video files
│   ├── processed/        # Preprocessed landmarks
│   └── annotations/      # JSON annotation files
├── models/
│   ├── sign_recognition/ # New spatial-temporal models
│   └── checkpoints/      # Trained models
├── scripts/
│   ├── data_collection/  # Recording tools
│   ├── data_preprocessing/ # Video processing
│   └── training/         # Training scripts
└── backend/
    └── ml_serving/       # Model loading & inference
```

## 🎓 Example Workflow

1. **Record 10 signs** (HELLO, THANK_YOU, PLEASE, etc.)
2. **Create annotations** JSON file
3. **Preprocess** videos to extract landmarks
4. **Train** spatial-temporal model
5. **Deploy** - backend auto-detects new model
6. **Test** - use frontend to test predictions

## 💡 Tips

- Start with 5-10 signs to test the pipeline
- Record each sign 5-10 times for better training
- Use consistent lighting and background
- Keep hands fully visible in frame
- Record at 30 FPS minimum

## 🐛 Troubleshooting

**MediaPipe not working?**
- It's optional! The system works without it
- For hand detection, download model from MediaPipe

**Model not loading?**
- Check `models/checkpoints/` directory
- Ensure checkpoint file exists
- Check console for error messages

**Training errors?**
- Ensure you have processed data files (.pkl)
- Check that annotation files match video paths
- Verify PyTorch Lightning is installed

## 📚 Next: Research Opportunities

Once you have a working enhanced model:
1. **Few-shot adaptation**: Personalize for new signers
2. **Cross-lingual transfer**: Adapt ASL model to ISL
3. **Grammar modeling**: Add sequence-to-sequence translation
4. **Fairness evaluation**: Test across demographics


