# SignBridge Datasets

This directory contains all datasets used for training and evaluating SignBridge models.

## Directory Structure

```
data/
├── raw/                    # Raw, unprocessed datasets
│   ├── asl/                # American Sign Language
│   ├── isl/                # Indian Sign Language
│   ├── gsl/                # German Sign Language
│   ├── bsl/                # British Sign Language (future)
│   └── reference_images/   # Reference alphabet images
├── processed/              # Preprocessed features
│   ├── landmarks/          # MediaPipe landmarks
│   ├── features/           # Engineered features
│   └── augmented/          # Augmented data
├── annotations/            # Annotation files
├── datasets/               # Ready-to-use train/val/test splits
└── sign_dictionary/        # Sign glosses for generation
```

## Datasets Overview

### 1. American Sign Language (ASL)

#### Sign MNIST Dataset
- **Location**: `raw/asl/mnist/`
- **Description**: Static images of ASL alphabet and digits
- **Training Samples**: 27,456
- **Test Samples**: 7,173
- **Classes**: 24 (A-Z excluding J and Z which require motion)
- **Format**: CSV (pixel values) + label
- **Image Size**: 28×28 grayscale
- **Source**: Kaggle Sign Language MNIST

**Usage Example:**
```python
import pandas as pd
train_df = pd.read_csv('data/raw/asl/mnist/train/sign_mnist_train.csv')
test_df = pd.read_csv('data/raw/asl/mnist/test/sign_mnist_test.csv')
```

#### Custom ASL Dataset
- **Location**: `raw/asl/custom_original/`
- **Description**: Multiple hand orientations and positions
- **Classes**: 36 (0-9, a-z)
- **Samples per class**: ~70 images
- **Variations**: 5 orientations (top, bottom, left, right, different)
- **Hands**: Multiple hand samples (hand1-5)
- **Format**: JPEG images in class directories

**Directory Structure:**
```
custom_original/
├── 0/ through 9/
├── a/ through z/
└── asl_dataset/ (nested copy - may need cleanup)
```

### 2. Indian Sign Language (ISL)

#### Custom ISL Dataset
- **Location**: `raw/isl/custom/`
- **Description**: Custom collected ISL alphabet signs
- **Classes**: A-Q visible (more may exist)
- **Samples per class**: ~100 images
- **Format**: JPG images numbered 0-99 per class
- **Collection Method**: Custom recording sessions

**Directory Structure:**
```
custom/
├── A/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ... (up to 99.jpg)
├── B/ through Q/
└── ...
```

#### Indian Dataset (Additional)
- **Location**: `raw/isl/indian/`
- **Status**: To be documented after inspection
- **Description**: Additional Indian sign language data

### 3. German Sign Language (GSL)

- **Location**: `raw/gsl/`
- **Main File**: `data.csv` (8.9 MB)
- **Reference**: `alphabet.png` (visual reference for German alphabet)
- **Description**: CSV dataset for German Sign Language
- **Format**: Requires inspection to determine structure

### 4. Reference Images

- **Location**: `raw/reference_images/`
- **Purpose**: Visual references for sign language alphabets
- **Files**:
  - `american_sign_language.PNG` - ASL alphabet chart (208 KB)
  - `amer_sign2.png` - Additional ASL reference (488 KB)
  - `amer_sign3.png` - ASL reference (45 KB)
- **Backgrounds**: `backgrounds/fondo_blanco/` - White background images

## Data Processing Pipeline

### 1. Raw Data → Landmarks

Extract MediaPipe landmarks from videos/images:

```bash
python scripts/data_preprocessing/landmark_extractor.py \
  --input data/raw/asl/custom/ \
  --output data/processed/landmarks/asl/
```

**Output**: `.npy` files with shape `(num_frames, 408)` containing:
- Pose landmarks: 33 points × 4 (x, y, z, visibility) = 132 features
- Left hand: 21 points × 3 (x, y, z) = 63 features
- Right hand: 21 points × 3 (x, y, z) = 63 features
- Face (key points): 50 points × 3 (x, y, z) = 150 features

### 2. Landmarks → Engineered Features

Extract higher-level features:

```bash
python scripts/data_preprocessing/feature_engineering.py \
  --input data/processed/landmarks/asl/ \
  --output data/processed/features/asl/
```

**Features**:
- Hand shape descriptors
- Motion features (velocity, acceleration)
- Spatial relationships (hand-face distance, etc.)

### 3. Feature Engineering → Augmentation

Apply data augmentation:

```bash
python scripts/data_preprocessing/augmentation.py \
  --input data/processed/features/asl/ \
  --output data/processed/augmented/asl/ \
  --augment_factor 3
```

**Augmentation Techniques**:
- Brightness/contrast adjustment
- Gaussian noise
- Motion blur
- Speed variation (for videos)
- Random rotations (preserving landmarks)

### 4. Create Train/Val/Test Splits

```bash
python scripts/data_preprocessing/dataset_builder.py \
  --input data/processed/augmented/asl/ \
  --output data/datasets/ \
  --split 0.7 0.15 0.15
```

**Output**: Pickled datasets ready for training

## Dataset Statistics

### Current Dataset Sizes

| Dataset | Classes | Train Samples | Test Samples | Total |
|---------|---------|---------------|--------------|-------|
| ASL MNIST | 24 | 27,456 | 7,173 | 34,629 |
| ASL Custom | 36 | ~2,520 | TBD | ~2,520 |
| ISL Custom | 17+ | ~1,700+ | TBD | ~1,700+ |
| GSL | TBD | TBD | TBD | TBD |

### Class Distribution

Check class balance:

```bash
python scripts/analysis/dataset_report.py \
  --input data/raw/asl/mnist/ \
  --output data/dataset_report.html
```

## Data Annotation Format

### Video Annotations (JSON)

```json
{
  "video_path": "path/to/video.mp4",
  "sign_label": "HELLO",
  "gloss": "HELLO",
  "start_frame": 0,
  "end_frame": 150,
  "signer_id": "signer_001",
  "language": "ASL",
  "dialect": "standard",
  "metadata": {
    "fps": 30,
    "resolution": [1280, 720],
    "handedness": "right",
    "lighting": "good"
  }
}
```

### Image Annotations (JSON)

```json
{
  "image_path": "path/to/image.jpg",
  "sign_label": "A",
  "signer_id": "signer_001",
  "language": "ASL",
  "hand_position": "top"
}
```

## Adding New Datasets

### 1. Public Dataset

```bash
# Download using provided scripts
python scripts/data_collection/download_wlasl.py --output data/raw/asl/wlasl/
python scripts/data_collection/download_how2sign.py --output data/raw/asl/how2sign/
```

### 2. Custom Recording

```bash
# Use recording tool
python scripts/data_collection/video_recorder.py \
  --output data/raw/asl/custom/ \
  --signer_id your_id \
  --signs HELLO THANK_YOU PLEASE
```

### 3. Validation

Always validate new data:

```bash
python scripts/data_collection/dataset_validator.py \
  --input data/raw/asl/custom/
```

## Data Quality Guidelines

### Video Requirements
- **Resolution**: Minimum 640×480, recommended 1280×720
- **FPS**: Minimum 30 FPS
- **Duration**: 2-10 seconds per sign
- **Lighting**: Even, well-lit
- **Background**: Plain, contrasting with skin tone
- **Framing**: Full upper body visible, hands always in frame

### Image Requirements
- **Resolution**: Minimum 224×224
- **Format**: JPG or PNG
- **Quality**: No blur, clear hand shapes
- **Consistency**: Similar framing across dataset

### Diversity Requirements
- **Signers**: At least 10 different signers per sign
- **Demographics**: Varied age, gender, ethnicity
- **Conditions**: Different lighting, backgrounds
- **Styles**: Different signing speeds and styles

## Known Issues

### ASL Custom Dataset
- ⚠️ Nested directory structure: `asl_dataset/asl_dataset/` may contain duplicates
- **Action**: Verify and remove duplicates

### ISL Custom Dataset
- ⚠️ Only classes A-Q are visible, may be incomplete
- **Action**: Verify complete alphabet coverage

### German Sign Language
- ⚠️ CSV structure not yet documented
- **Action**: Inspect and document format

## Data Storage Best Practices

### Version Control
- **Raw data**: Do NOT commit to git (add to .gitignore)
- **Processed data**: Optionally commit small processed files
- **Annotations**: DO commit (small JSON files)
- **Models**: Use Git LFS or separate storage

### Backup Strategy
1. Raw data: Cloud storage (S3, Google Drive)
2. Processed data: Can be regenerated from raw
3. Annotations: Git repository (backed up)
4. Regular backups before major changes

### Storage Optimization
```bash
# Compress old datasets
tar -czf data/archive/asl_mnist_v1.tar.gz data/raw/asl/mnist/

# Clean up intermediate files
rm -rf data/processed/temp/
```

## Citations

### Sign MNIST
```bibtex
@misc{signmnist2018,
  title={Sign Language MNIST},
  author={Tecperson},
  year={2018},
  url={https://www.kaggle.com/datamunge/sign-language-mnist}
}
```

### Other Datasets
- WLASL: [Citation needed when implemented]
- How2Sign: [Citation needed when implemented]
- MS-ASL: [Citation needed when implemented]

## Support

For dataset-related issues:
1. Check dataset statistics: `python scripts/analysis/dataset_report.py`
2. Validate data: `python scripts/data_collection/dataset_validator.py`
3. Report issues: [GitHub Issues](https://github.com/yourusername/SignBridge/issues)

---

Last Updated: 2024-01-20
