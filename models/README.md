# SignBridge Models

This directory contains all machine learning models and training code for SignBridge.

## Directory Structure

```
models/
├── sign_recognition/       # Sign → Text models
│   ├── spatial_encoder.py
│   ├── temporal_encoder.py
│   ├── classifier.py
│   ├── model.py
│   ├── dataset.py
│   └── lightning_module.py
├── sign_generation/        # Text → Sign models
│   ├── text_processor.py
│   ├── gloss_to_motion.py
│   └── motion_smoother.py
├── shared/                 # Shared components
│   ├── feature_extractor.py
│   ├── positional_encoding.py
│   └── utils.py
├── checkpoints/            # Model checkpoints (.gitignored)
│   └── .gitkeep
└── evaluation/             # Evaluation metrics
    └── metrics.py
```

## Model Architectures

### 1. Sign Recognition Model (Sign → Text)

**Architecture**: SpatialEncoder → TemporalEncoder → Classifier

#### Components:

**SpatialEncoder** (`sign_recognition/spatial_encoder.py`)
- Processes individual frames
- Input: (batch, seq_len, 408) landmarks
- Architecture: 2-layer MLP with BatchNorm and Dropout
- Output: (batch, seq_len, 256) spatial features

**TemporalEncoder** (`sign_recognition/temporal_encoder.py`)
- Models temporal dynamics
- Input: (batch, seq_len, 256) spatial features
- Architecture: Transformer Encoder (4 layers, 8 heads)
- Positional encoding for sequence order
- Output: (batch, seq_len, 256) temporal features

**Classifier** (`sign_recognition/classifier.py`)
- Maps features to sign classes
- Input: (batch, 256) pooled features
- Architecture: 2-layer MLP
- Output: (batch, num_classes) logits

**Complete Model** (`sign_recognition/model.py`)
```python
from models.sign_recognition.model import SignRecognitionModel

model = SignRecognitionModel(
    input_dim=408,          # MediaPipe landmarks
    spatial_dim=256,
    temporal_dim=256,
    num_classes=1000,       # Number of signs
    num_transformer_layers=4
)
```

#### Feature Dimensions:

Input landmarks (408 dimensions):
- Pose: 33 points × 4 (x, y, z, visibility) = 132
- Left hand: 21 points × 3 (x, y, z) = 63
- Right hand: 21 points × 3 (x, y, z) = 63
- Face (key): 50 points × 3 (x, y, z) = 150

### 2. Sign Generation Model (Text → Sign)

**Pipeline**: Text → NLP Processing → Gloss Generation → Motion Synthesis → Avatar Animation

#### Components:

**TextProcessor** (`sign_generation/text_processor.py`)
- Converts natural language text to sign glosses
- Uses fine-tuned BERT/T5 for translation
- Handles grammar transformation (spoken → sign)
- Removes articles, adjusts word order

**GlossToMotion** (`sign_generation/gloss_to_motion.py`)
- Maps glosses to motion sequences
- Loads sign dictionary (gloss → landmarks)
- Interpolates between signs
- Adds facial expressions

**MotionSmoother** (`sign_generation/motion_smoother.py`)
- Smooths generated motion
- Uses Savitzky-Golay filter
- Bezier curve interpolation
- Ensures natural transitions

## Training

### Quick Start

```bash
# Train sign recognition model
python scripts/training/train_model.py \
    --config configs/train_config.yaml \
    --data data/processed/asl/ \
    --output models/checkpoints/
```

### Using PyTorch Lightning

```python
from models.sign_recognition.lightning_module import SignRecognitionLightning
from pytorch_lightning import Trainer

# Initialize model
model = SignRecognitionLightning(
    num_classes=100,
    learning_rate=1e-4
)

# Train
trainer = Trainer(
    max_epochs=100,
    accelerator='gpu',
    devices=1,
    precision=16
)
trainer.fit(model, train_loader, val_loader)
```

### Hyperparameters

Default configuration:
- **Batch size**: 32
- **Learning rate**: 1e-4
- **Optimizer**: AdamW (weight_decay=0.01)
- **Scheduler**: ReduceLROnPlateau
- **Loss**: CrossEntropyLoss with label smoothing (0.1)
- **Gradient clipping**: max_norm=1.0
- **Early stopping**: patience=10

## Evaluation

### Metrics

**Classification Metrics** (`evaluation/metrics.py`):
- Top-1 and Top-5 accuracy
- Per-class precision, recall, F1-score
- Confusion matrix
- Per-signer performance analysis

**Translation Metrics**:
- BLEU score (sign-to-text)
- Human evaluation (text-to-sign naturalness)
- Grammar accuracy
- Facial expression correctness

### Evaluation Script

```bash
python scripts/training/evaluate_model.py \
    --checkpoint models/checkpoints/best_model.ckpt \
    --test_data data/datasets/test/ \
    --output results/evaluation_report.html
```

## Model Checkpoints

### Checkpoint Format

```python
checkpoint = {
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'train_loss': float,
    'val_loss': float,
    'val_accuracy': float,
    'hyperparameters': dict,
    'timestamp': str
}
```

### Loading Checkpoints

```python
from models.sign_recognition.model import SignRecognitionModel
import torch

# Load model
model = SignRecognitionModel.load_from_checkpoint(
    'models/checkpoints/best_model.ckpt'
)
model.eval()

# Inference
with torch.no_grad():
    output = model(input_tensor)
```

### Checkpoint Management

Checkpoints are saved in `models/checkpoints/` with naming:
```
sign-recognition-epoch={epoch:02d}-val_acc={val_acc:.2f}.ckpt
```

Best models saved based on validation accuracy.

## Model Export

### ONNX Export

```python
import torch

model = SignRecognitionModel(...)
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

dummy_input = torch.randn(1, 150, 408)
torch.onnx.export(
    model,
    dummy_input,
    'models/exported/model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch', 1: 'sequence'}}
)
```

### TorchScript Export

```python
model = SignRecognitionModel(...)
model.eval()

scripted = torch.jit.script(model)
scripted.save('models/exported/model.pt')
```

## Performance Optimization

### Model Quantization

```python
import torch.quantization

# Post-training quantization
model_fp32 = SignRecognitionModel(...)
model_fp32.eval()

model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

### Mixed Precision Training

```python
trainer = Trainer(
    precision=16,  # Use FP16
    accelerator='gpu'
)
```

### Batch Inference

```python
# Process multiple videos at once
batch = torch.stack([video1, video2, video3])
outputs = model(batch)  # Faster than individual inference
```

## Model Versioning

### Version Naming Convention

```
v{major}.{minor}.{patch}-{descriptor}

Examples:
- v1.0.0-baseline
- v1.1.0-improved-temporal
- v2.0.0-multimodal
```

### Model Registry

Track models in `models/registry.json`:

```json
{
  "v1.0.0-baseline": {
    "checkpoint": "checkpoints/v1.0.0/best_model.ckpt",
    "accuracy": 0.82,
    "num_classes": 100,
    "trained_on": "2024-01-20",
    "description": "Initial baseline model"
  }
}
```

## Research Models

### Novel Contributions

1. **Few-Shot Adaptation**
   - Meta-learning for signer personalization
   - Location: `models/research/few_shot/`

2. **Cross-Lingual Transfer**
   - Multi-task learning across sign languages
   - Location: `models/research/cross_lingual/`

3. **Grammar-Aware Translation**
   - Structured prediction with linguistic constraints
   - Location: `models/research/grammar_aware/`

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size
batch_size = 16  # or 8

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use gradient accumulation
trainer = Trainer(accumulate_grad_batches=2)
```

**2. Model Not Converging**
```python
# Reduce learning rate
learning_rate = 1e-5

# Add gradient clipping
trainer = Trainer(gradient_clip_val=1.0)

# Check for NaN in data
torch.isnan(tensor).any()
```

**3. Slow Training**
```python
# Use DataLoader with multiple workers
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True
)

# Enable cudnn benchmark
torch.backends.cudnn.benchmark = True
```

## References

### Papers
- Transformer architecture: Vaswani et al. "Attention Is All You Need"
- Sign language recognition: [Add relevant papers]
- Temporal modeling: [Add relevant papers]

### Code Examples
- PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/
- Transformers: https://huggingface.co/docs/transformers/

## Support

For model-related questions:
1. Check model architecture in code
2. Review training logs in `logs/`
3. Inspect checkpoints in `models/checkpoints/`
4. Open issue on GitHub

---

Last Updated: 2024-01-20
