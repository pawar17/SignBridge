# SignBridge - Complete Project Reference

**Last Updated**: January 20, 2024
**Status**: ✅ Ready for Development
**Version**: 0.1.0

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Current Status](#current-status)
3. [Directory Structure](#directory-structure)
4. [Dataset Information](#dataset-information)
5. [Configuration](#configuration)
6. [Development Workflow](#development-workflow)
7. [Available Commands](#available-commands)
8. [Progress Tracking](#progress-tracking)
9. [Next Steps](#next-steps)
10. [Troubleshooting](#troubleshooting)
11. [Quick Reference](#quick-reference)

---

## Project Overview

### What is SignBridge?

SignBridge is an AI-powered platform for bidirectional sign language translation and learning:
- **Sign → Text/Speech**: Real-time translation using computer vision
- **Text/Speech → Sign**: Avatar-based sign generation
- **Learning Platform**: Adaptive curriculum for learning sign languages
- **Multi-Language**: ASL, ISL, BSL, JSL, LSF, GSL support

### Key Features
- Bidirectional translation with cultural context
- Multi-language support with dialect recognition
- Context-aware processing (grammar preservation)
- Privacy-first (on-device processing option)
- Accessible design (WCAG 2.2 AAA)

### Technology Stack
- **ML/AI**: PyTorch, MediaPipe, Transformers, TensorFlow.js
- **Backend**: FastAPI, PostgreSQL, Redis, Celery
- **Frontend**: React, Three.js, Material-UI
- **Infrastructure**: Docker, Kubernetes, AWS/GCP

---

## Current Status

### ✅ Completed (January 20, 2024)

**Organization**
- [x] Professional directory structure (40+ directories)
- [x] All datasets organized (~38,849+ samples)
- [x] Documentation consolidated
- [x] Python modules with __init__.py files (20 files)
- [x] Empty directories marked with .gitkeep (27 files)

**Configuration**
- [x] .gitignore configured for ML projects
- [x] requirements.txt (70+ dependencies)
- [x] .env.example (133 environment variables)
- [x] setup.py (package configuration)
- [x] Docker Compose (8 services)

**Core Code**
- [x] Configuration system (backend/config.py)
- [x] Logging system (backend/utils/logger.py)
- [x] Landmark extractor (scripts/data_preprocessing/landmark_extractor.py)

**Infrastructure**
- [x] Docker Compose with 8 services
- [x] PostgreSQL, Redis, Celery configured
- [x] Backend and frontend Dockerfiles

### 🔄 In Progress

- [ ] Create .env file from template
- [ ] Install Python dependencies
- [ ] Test landmark extraction on all datasets
- [ ] Build dataset builder script

### 📋 Next Phase

**Week 1-2: Data Pipeline**
- [ ] Extract landmarks from all datasets
- [ ] Create train/val/test splits
- [ ] Generate dataset statistics

**Week 3-4: Model Development**
- [ ] Implement model architecture
- [ ] Set up training pipeline
- [ ] Train baseline model

**Week 5-6: Backend API**
- [ ] Create FastAPI application
- [ ] Implement core endpoints
- [ ] Add authentication

---

## Directory Structure

```
SignBridge/
│
├── 📚 docs/                          # All documentation
│   ├── PRD.md                        # Product Requirements Document
│   ├── implementation_guide.md      # Technical implementation guide
│   ├── claude_code_guide.md         # Claude Code specific guide
│   ├── api/                          # API documentation
│   ├── architecture/                 # Architecture diagrams
│   └── user_guide/                   # User documentation
│
├── 📊 data/                          # All datasets (~38,849+ samples)
│   │
│   ├── raw/                          # Raw, unprocessed data
│   │   ├── asl/                      # American Sign Language
│   │   │   ├── mnist/                # Sign MNIST (34,629 samples)
│   │   │   │   ├── train/            # 27,456 training samples
│   │   │   │   │   ├── sign_mnist_train.csv
│   │   │   │   │   └── images/
│   │   │   │   └── test/             # 7,173 test samples
│   │   │   │       ├── sign_mnist_test.csv
│   │   │   │       └── images/
│   │   │   └── custom_original/      # Custom ASL (~2,520 images)
│   │   │       ├── 0/ through 9/     # Digits
│   │   │       └── a/ through z/     # Letters
│   │   │
│   │   ├── isl/                      # Indian Sign Language
│   │   │   ├── custom/               # ISL alphabet (~1,700+ images)
│   │   │   │   └── A/ through Q+/    # Alphabet
│   │   │   └── numbers_and_letters/  # Additional ISL data
│   │   │       ├── 1/ through 9/
│   │   │       └── A/ through H+/
│   │   │
│   │   ├── gsl/                      # German Sign Language
│   │   │   ├── data.csv              # CSV dataset (8.9 MB)
│   │   │   └── alphabet.png          # Reference alphabet
│   │   │
│   │   ├── bsl/                      # British Sign Language (future)
│   │   │
│   │   └── reference_images/         # Reference alphabet charts
│   │       ├── american_sign_language.PNG
│   │       ├── amer_sign2.png
│   │       ├── amer_sign3.png
│   │       └── backgrounds/
│   │           └── fondo_blanco/
│   │
│   ├── processed/                    # Preprocessed features
│   │   ├── landmarks/                # MediaPipe landmarks (.npy)
│   │   ├── features/                 # Engineered features
│   │   └── augmented/                # Augmented data
│   │
│   ├── annotations/                  # Annotation JSON files
│   │   ├── train_annotations.json
│   │   ├── val_annotations.json
│   │   └── test_annotations.json
│   │
│   ├── datasets/                     # Ready-to-use train/val/test
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   └── sign_dictionary/              # Sign glosses for generation
│       └── glosses/
│
├── 🧠 models/                        # ML models and training
│   ├── sign_recognition/             # Sign → Text models
│   │   ├── spatial_encoder.py
│   │   ├── temporal_encoder.py
│   │   ├── classifier.py
│   │   ├── model.py
│   │   ├── dataset.py
│   │   └── lightning_module.py
│   │
│   ├── sign_generation/              # Text → Sign models
│   │   ├── text_processor.py
│   │   ├── gloss_to_motion.py
│   │   └── motion_smoother.py
│   │
│   ├── shared/                       # Shared components
│   │   ├── feature_extractor.py
│   │   ├── positional_encoding.py
│   │   └── utils.py
│   │
│   ├── checkpoints/                  # Model weights (.gitignored)
│   │   └── .gitkeep
│   │
│   └── evaluation/                   # Evaluation metrics
│       └── metrics.py
│
├── 🔧 backend/                       # FastAPI backend
│   ├── config.py                     # ✅ Configuration (Pydantic)
│   │
│   ├── api/
│   │   ├── main.py                   # FastAPI app
│   │   ├── routes/                   # API endpoints
│   │   │   ├── translation.py
│   │   │   ├── learning.py
│   │   │   ├── user.py
│   │   │   └── analytics.py
│   │   ├── schemas/                  # Pydantic schemas
│   │   │   ├── translation_schemas.py
│   │   │   ├── user_schemas.py
│   │   │   └── learning_schemas.py
│   │   └── dependencies.py
│   │
│   ├── services/                     # Business logic
│   │   ├── translation_service.py
│   │   ├── learning_engine.py
│   │   ├── user_service.py
│   │   └── celery_app.py
│   │
│   ├── database/                     # Database layer
│   │   ├── models.py                 # SQLAlchemy ORM
│   │   ├── crud.py                   # CRUD operations
│   │   └── database.py               # DB connection
│   │
│   ├── ml_serving/                   # Model inference
│   │   ├── model_loader.py
│   │   ├── inference_pipeline.py
│   │   └── cache.py
│   │
│   ├── middleware/                   # Middleware
│   │   ├── auth.py                   # JWT authentication
│   │   └── rate_limit.py             # Rate limiting
│   │
│   └── utils/
│       └── logger.py                 # ✅ Logging system
│
├── 🎨 frontend/                      # React frontend
│   ├── public/                       # Static files
│   ├── src/
│   │   ├── components/               # React components
│   │   │   ├── Layout/
│   │   │   ├── Translation/
│   │   │   ├── Learning/
│   │   │   ├── Avatar/
│   │   │   └── Common/
│   │   ├── hooks/                    # Custom React hooks
│   │   ├── services/                 # API clients
│   │   ├── utils/                    # Utilities
│   │   └── App.tsx
│   ├── package.json
│   └── tsconfig.json
│
├── 📓 notebooks/                     # Jupyter notebooks
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
│
├── 🔨 scripts/                       # Utility scripts
│   ├── data_collection/
│   │   ├── video_recorder.py
│   │   ├── download_datasets.py
│   │   └── dataset_validator.py
│   │
│   ├── data_preprocessing/
│   │   ├── landmark_extractor.py     # ✅ MediaPipe extraction
│   │   ├── feature_engineering.py
│   │   ├── augmentation.py
│   │   └── dataset_builder.py
│   │
│   ├── training/
│   │   ├── train_model.py
│   │   ├── evaluate_model.py
│   │   └── tune_hyperparameters.py
│   │
│   ├── deployment/
│   │   └── deploy.sh
│   │
│   └── analysis/
│       └── dataset_report.py
│
├── 🧪 tests/                         # Tests
│   ├── test_models.py
│   ├── test_api.py
│   ├── test_frontend/
│   └── integration/
│
├── 🐳 docker/                        # Docker configuration
│   ├── docker-compose.yml            # ✅ 8 services configured
│   ├── Dockerfile.backend            # ✅ Python backend
│   └── Dockerfile.frontend           # ✅ Node.js frontend
│
├── 🔄 .github/                       # GitHub workflows
│   └── workflows/
│       ├── test.yml
│       └── deploy.yml
│
├── 📄 Project Files
│   ├── README.md                     # Project overview
│   ├── PROJECT_REFERENCE.md          # 📍 THIS FILE
│   ├── requirements.txt              # Python dependencies
│   ├── setup.py                      # Package setup
│   ├── .env.example                  # Environment template
│   ├── .gitignore                    # Git ignore rules
│   └── logs/                         # Application logs
│
└── 📊 Statistics
    ├── Directories: 40+
    ├── Python Files: 23
    ├── __init__.py: 20
    ├── .gitkeep: 27
    └── Total Samples: ~38,849+
```

---

## Dataset Information

### Summary Table

| Dataset | Type | Classes | Train | Test | Total | Location | Status |
|---------|------|---------|-------|------|-------|----------|--------|
| ASL MNIST | Images (28×28) | 24 | 27,456 | 7,173 | 34,629 | `data/raw/asl/mnist/` | ✅ Ready |
| ASL Custom | Images | 36 | ~2,520 | TBD | ~2,520 | `data/raw/asl/custom_original/` | ✅ Ready |
| ISL Custom | Images | 17+ | ~1,700+ | TBD | ~1,700+ | `data/raw/isl/custom/` | ✅ Ready |
| ISL Numbers | Images | Various | TBD | TBD | TBD | `data/raw/isl/numbers_and_letters/` | ✅ Ready |
| GSL | CSV | TBD | TBD | TBD | 8.9MB | `data/raw/gsl/` | ✅ Ready |

**Total: ~38,849+ samples across all datasets**

### Dataset Details

#### 1. ASL MNIST (34,629 samples)
- **Location**: `data/raw/asl/mnist/`
- **Format**: CSV with pixel values (28×28 grayscale)
- **Classes**: 24 letters (A-Z excluding J and Z)
- **Train**: 27,456 samples in `train/sign_mnist_train.csv`
- **Test**: 7,173 samples in `test/sign_mnist_test.csv`
- **Usage**: Baseline model training
- **Citation**: Kaggle Sign Language MNIST

#### 2. ASL Custom Dataset (~2,520 images)
- **Location**: `data/raw/asl/custom_original/`
- **Format**: JPEG images in class directories
- **Classes**: 36 (0-9, a-z)
- **Variations**: 5 orientations per hand (top, bottom, left, right, different)
- **Hands**: Multiple hand samples (hand1-5)
- **Samples per class**: ~70 images
- **Usage**: Model improvement, augmentation testing

#### 3. ISL Custom Dataset (~1,700+ images)
- **Location**: `data/raw/isl/custom/`
- **Format**: JPG images numbered 0-99 per class
- **Classes**: A-Q+ (17+ letters)
- **Samples per class**: ~100 images
- **Usage**: ISL recognition model

#### 4. ISL Numbers and Letters
- **Location**: `data/raw/isl/numbers_and_letters/`
- **Format**: JPG images in class directories
- **Classes**: Numbers 1-9, Letters A-H+
- **Usage**: Additional ISL training data

#### 5. German Sign Language (8.9 MB)
- **Location**: `data/raw/gsl/`
- **Format**: CSV file
- **Reference**: `alphabet.png` (visual reference)
- **Status**: Needs inspection to determine structure
- **Usage**: Future GSL support

### Data Processing Pipeline

```
Raw Data → Landmarks → Features → Datasets → Training
```

1. **Raw Data** (`data/raw/`)
   - Original videos/images
   - Reference alphabet charts

2. **Landmarks** (`data/processed/landmarks/`)
   - MediaPipe extracted landmarks (408 features/frame)
   - Format: .npy files (num_frames, 408)
   - Extract with: `landmark_extractor.py`

3. **Features** (`data/processed/features/`)
   - Engineered features (hand shape, motion, spatial)
   - Velocity, acceleration, relationships

4. **Datasets** (`data/datasets/`)
   - Train/Val/Test splits (70/15/15)
   - Pickled format for fast loading

### Landmark Format (408 features)

```python
# Per frame: 408 features
- Pose: 33 points × 4 (x, y, z, visibility) = 132
- Left Hand: 21 points × 3 (x, y, z) = 63
- Right Hand: 21 points × 3 (x, y, z) = 63
- Face (key points): 50 points × 3 (x, y, z) = 150
```

---

## Configuration

### Environment Variables (.env)

**Create from template:**
```bash
cp .env.example .env
# Edit .env with your settings
```

**Key Variables:**

```bash
# Application
APP_ENV=development              # development, staging, production
DEBUG=True

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/signbridge

# Redis
REDIS_URL=redis://localhost:6379/0

# ML Settings
ML_DEVICE=cuda                   # cuda or cpu
MODEL_CHECKPOINT_DIR=models/checkpoints

# Security
SECRET_KEY=<generate-random-key>

# Languages
LANGUAGES_ENABLED=ASL,ISL,BSL,JSL,LSF,GSL
```

### Python Dependencies

**Install:**
```bash
pip install -r requirements.txt
```

**Key Packages (70+ total):**
- PyTorch, TensorFlow
- MediaPipe, OpenCV
- FastAPI, Uvicorn
- SQLAlchemy, Redis
- Celery, Flower
- PyTorch Lightning
- Transformers, Datasets

### Docker Services

**8 Services Configured:**

| Service | Port | Description |
|---------|------|-------------|
| postgres | 5432 | PostgreSQL database |
| redis | 6379 | Cache & message broker |
| backend | 8000 | FastAPI application |
| celery-worker | - | Async task processing |
| celery-beat | - | Scheduled tasks |
| flower | 5555 | Celery monitoring |
| frontend | 3000 | React application |
| pgadmin | 5050 | Database admin UI |

**Start Services:**
```bash
cd docker
docker-compose up -d
```

---

## Development Workflow

### Phase 1: Data Preprocessing (Current)

**Goal**: Extract landmarks and prepare datasets

1. **Extract Landmarks from ASL MNIST**
   ```bash
   python scripts/data_preprocessing/landmark_extractor.py \
       --input data/raw/asl/mnist/train/images/ \
       --output data/processed/landmarks/asl_mnist/train/ \
       --extension .jpg
   ```

2. **Extract Landmarks from Custom ASL**
   ```bash
   python scripts/data_preprocessing/landmark_extractor.py \
       --input data/raw/asl/custom_original/ \
       --output data/processed/landmarks/asl_custom/ \
       --extension .jpeg
   ```

3. **Extract Landmarks from ISL**
   ```bash
   python scripts/data_preprocessing/landmark_extractor.py \
       --input data/raw/isl/custom/ \
       --output data/processed/landmarks/isl_custom/ \
       --extension .jpg
   ```

4. **Build Dataset**
   ```bash
   # TODO: Create dataset_builder.py
   python scripts/data_preprocessing/dataset_builder.py \
       --input data/processed/landmarks/ \
       --output data/datasets/ \
       --split 0.7 0.15 0.15
   ```

### Phase 2: Model Development

**Goal**: Train baseline sign recognition model

1. **Implement Model Architecture**
   - Create `models/sign_recognition/spatial_encoder.py`
   - Create `models/sign_recognition/temporal_encoder.py`
   - Create `models/sign_recognition/model.py`

2. **Set Up Training**
   - Create `scripts/training/train_model.py`
   - Configure PyTorch Lightning
   - Add TensorBoard/Wandb logging

3. **Train Model**
   ```bash
   python scripts/training/train_model.py \
       --config configs/train_config.yaml \
       --data data/datasets/ \
       --output models/checkpoints/
   ```

### Phase 3: Backend API

**Goal**: Create FastAPI backend with ML serving

1. **Set Up FastAPI**
   - Create `backend/api/main.py`
   - Define routes in `backend/api/routes/`
   - Create schemas in `backend/api/schemas/`

2. **Model Serving**
   - Create `backend/ml_serving/model_loader.py`
   - Create `backend/ml_serving/inference_pipeline.py`

3. **Run Backend**
   ```bash
   uvicorn backend.api.main:app --reload
   ```

### Phase 4: Frontend

**Goal**: Create React translation interface

1. **Initialize React App**
   ```bash
   cd frontend
   npx create-react-app . --template typescript
   npm install
   ```

2. **Create Components**
   - Translation interface
   - Video capture with MediaPipe
   - 3D avatar renderer (Three.js)

3. **Run Frontend**
   ```bash
   npm start
   ```

---

## Available Commands

### Data Processing

```bash
# Extract landmarks
python scripts/data_preprocessing/landmark_extractor.py \
    --input <input_dir> \
    --output <output_dir> \
    --extension <.jpg|.jpeg|.mp4> \
    --model-complexity 2 \
    --min-confidence 0.5

# Build datasets (TODO: create this script)
python scripts/data_preprocessing/dataset_builder.py \
    --input data/processed/landmarks/ \
    --output data/datasets/ \
    --split 0.7 0.15 0.15

# Validate datasets
python scripts/data_collection/dataset_validator.py \
    --input data/raw/asl/custom/
```

### Model Training

```bash
# Train model (TODO: create this script)
python scripts/training/train_model.py \
    --config configs/train_config.yaml \
    --data data/datasets/train/ \
    --output models/checkpoints/

# Evaluate model (TODO: create this script)
python scripts/training/evaluate_model.py \
    --checkpoint models/checkpoints/best.ckpt \
    --test-data data/datasets/test/

# Hyperparameter tuning (TODO: create this script)
python scripts/training/tune_hyperparameters.py \
    --trials 50
```

### Development

```bash
# Test configuration
python backend/config.py

# Test logger
python backend/utils/logger.py

# Run backend
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

# Run Celery worker
celery -A backend.services.celery_app worker --loglevel=info

# Run frontend
cd frontend && npm start
```

### Docker

```bash
# Start all services
docker-compose up -d

# View logs (all services)
docker-compose logs -f

# View logs (specific service)
docker-compose logs -f backend

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build

# Remove volumes (clean slate)
docker-compose down -v
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=models --cov=backend tests/

# Run specific test file
pytest tests/test_models.py -v

# Run specific test
pytest tests/test_models.py::test_model_forward -v
```

### Git

```bash
# Initialize git (if needed)
git init
git add .
git commit -m "Initial commit: Organized project structure"

# Check status
git status

# Add remote
git remote add origin <your-repo-url>
git push -u origin main
```

---

## Progress Tracking

### Development Phases

#### ✅ Phase 0: Setup (Completed - Jan 20, 2024)
- [x] Organize project structure
- [x] Move and categorize datasets
- [x] Create configuration files
- [x] Set up Docker infrastructure
- [x] Create core Python modules
- [x] Write comprehensive documentation

#### 🔄 Phase 1: Data Pipeline (In Progress)
**Timeline**: Week 1-2

- [ ] Create .env file and configure
- [ ] Install all dependencies
- [ ] Extract landmarks from ASL MNIST
- [ ] Extract landmarks from ASL Custom
- [ ] Extract landmarks from ISL datasets
- [ ] Create dataset builder script
- [ ] Generate train/val/test splits
- [ ] Create dataset statistics report
- [ ] Validate all processed data

**Deliverables**:
- Processed landmarks for all datasets
- Train/val/test splits ready for training
- Dataset report with statistics

#### 📋 Phase 2: Model Development (Week 3-4)
- [ ] Implement spatial encoder
- [ ] Implement temporal encoder
- [ ] Implement classifier
- [ ] Create complete model
- [ ] Create dataset loader
- [ ] Set up PyTorch Lightning module
- [ ] Configure training pipeline
- [ ] Train baseline model (target: >70% accuracy)
- [ ] Evaluate model performance
- [ ] Save best checkpoint

**Deliverables**:
- Working sign recognition model
- Training logs and metrics
- Evaluation report

#### 📋 Phase 3: Backend API (Week 5-6)
- [ ] Create FastAPI application
- [ ] Implement translation endpoints
- [ ] Add model serving infrastructure
- [ ] Set up database models
- [ ] Add authentication (JWT)
- [ ] Implement rate limiting
- [ ] Add caching layer
- [ ] Write API tests
- [ ] Create API documentation

**Deliverables**:
- Working API with translation endpoint
- API documentation (Swagger)
- Tests passing

#### 📋 Phase 4: Frontend (Week 7-8)
- [ ] Initialize React application
- [ ] Create component structure
- [ ] Implement video capture
- [ ] Integrate MediaPipe
- [ ] Create translation interface
- [ ] Build 3D avatar renderer
- [ ] Connect to backend API
- [ ] Add error handling
- [ ] Write frontend tests

**Deliverables**:
- Working web interface
- Real-time translation demo
- User documentation

### Current Week Tasks

**This Week (Week 1)**:
1. [ ] Copy `.env.example` to `.env` and configure
2. [ ] Install Python dependencies: `pip install -r requirements.txt`
3. [ ] Test landmark extraction on sample data
4. [ ] Extract landmarks from ASL MNIST train set
5. [ ] Extract landmarks from ASL MNIST test set
6. [ ] Document any issues encountered

**Next Week (Week 2)**:
1. [ ] Extract landmarks from all ASL Custom data
2. [ ] Extract landmarks from ISL data
3. [ ] Create dataset builder script
4. [ ] Generate train/val/test splits
5. [ ] Create dataset statistics report

### Metrics & Goals

**Data Processing**:
- Target: Process all ~38,849+ samples
- Landmark extraction success rate: >95%
- Processing speed: >10 images/second

**Model Performance** (Phase 2):
- Baseline accuracy: >70%
- Top-5 accuracy: >90%
- Inference speed: <100ms per sign

**API Performance** (Phase 3):
- Response time: <500ms (p95)
- Throughput: >100 requests/second
- Uptime: >99.9%

---

## Next Steps

### Immediate (Today)
1. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env:
   # - Set ML_DEVICE (cuda or cpu)
   # - Configure DATABASE_URL
   # - Generate SECRET_KEY
   ```

2. **Install Dependencies**
   ```bash
   conda create -n signbridge python=3.10
   conda activate signbridge
   pip install -r requirements.txt
   ```

3. **Test Setup**
   ```bash
   # Test configuration
   python backend/config.py

   # Test logger
   python backend/utils/logger.py

   # Test landmark extractor help
   python scripts/data_preprocessing/landmark_extractor.py --help
   ```

### This Week
4. **Extract Sample Landmarks**
   ```bash
   # Test on small dataset
   python scripts/data_preprocessing/landmark_extractor.py \
       --input data/raw/asl/custom_original/0/ \
       --output data/processed/landmarks/test/ \
       --extension .jpeg
   ```

5. **Start Processing Full Datasets**
   ```bash
   # ASL MNIST Train
   python scripts/data_preprocessing/landmark_extractor.py \
       --input data/raw/asl/mnist/train/images/ \
       --output data/processed/landmarks/asl_mnist/train/ \
       --extension .jpg
   ```

### Next Week
6. **Build Dataset Pipeline**
   - Create `dataset_builder.py`
   - Generate train/val/test splits
   - Create data loaders

7. **Start Model Development**
   - Implement model architecture
   - Set up training script
   - Configure experiments

---

## Troubleshooting

### Common Issues

#### 1. MediaPipe Installation Fails
```bash
# Try specific version
pip install mediapipe==0.10.9

# Or use conda
conda install -c conda-forge mediapipe
```

#### 2. CUDA Not Available
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Module Not Found Errors
```bash
# Make sure you're in project root
cd /path/to/SignBridge

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%cd%          # Windows CMD
$env:PYTHONPATH += ";$(pwd)"              # Windows PowerShell
```

#### 4. Database Connection Error
```bash
# Start PostgreSQL with Docker
docker-compose up -d postgres

# Or check local PostgreSQL
# Windows: net start postgresql
# Linux: sudo systemctl start postgresql

# Verify connection
psql -U signbridge_user -h localhost -d signbridge
```

#### 5. Port Already in Use
```bash
# Check what's using port
netstat -ano | findstr :8000    # Windows
lsof -i :8000                   # Linux/Mac

# Kill process or use different port in .env
API_PORT=8001
```

#### 6. Out of Memory During Processing
```bash
# Reduce batch size
ML_BATCH_SIZE=16  # in .env

# Or process in smaller chunks
python scripts/data_preprocessing/landmark_extractor.py \
    --input data/raw/asl/custom_original/a/ \
    --output data/processed/landmarks/asl_custom/a/
```

### Getting Help

1. Check this reference file for documentation
2. Review relevant README in subdirectories
3. Check logs in `logs/` directory
4. Review error messages carefully
5. Open GitHub issue with:
   - Error message
   - Steps to reproduce
   - Environment details (OS, Python version, etc.)

---

## Quick Reference

### File Locations

| What | Where |
|------|-------|
| **Main reference** | `PROJECT_REFERENCE.md` (this file) |
| **Product requirements** | `docs/PRD.md` |
| **Implementation guide** | `docs/implementation_guide.md` |
| **Dataset info** | `data/README.md` |
| **Model info** | `models/README.md` |
| **Environment config** | `.env` (create from `.env.example`) |
| **Dependencies** | `requirements.txt` |
| **Docker config** | `docker/docker-compose.yml` |

### Key Scripts

| Script | Purpose |
|--------|---------|
| `backend/config.py` | Load and validate configuration |
| `backend/utils/logger.py` | Logging utilities |
| `scripts/data_preprocessing/landmark_extractor.py` | Extract MediaPipe landmarks |
| `scripts/data_preprocessing/dataset_builder.py` | TODO: Build datasets |
| `scripts/training/train_model.py` | TODO: Train models |

### Quick Commands

```bash
# Setup
cp .env.example .env
pip install -r requirements.txt

# Test
python backend/config.py
python backend/utils/logger.py

# Extract landmarks
python scripts/data_preprocessing/landmark_extractor.py \
    --input data/raw/asl/custom_original/0/ \
    --output data/processed/landmarks/test/ \
    --extension .jpeg

# Docker
cd docker && docker-compose up -d

# Backend
uvicorn backend.api.main:app --reload

# Tests
pytest tests/ -v
```

### Project Stats

| Metric | Value |
|--------|-------|
| **Directories** | 40+ |
| **Python files** | 23 |
| **Data samples** | ~38,849+ |
| **Dependencies** | 70+ packages |
| **Docker services** | 8 |
| **Lines of documentation** | 1000+ |
| **Status** | ✅ Ready for Development |

---

## Update Log

### January 20, 2024
- ✅ Organized project structure
- ✅ Consolidated documentation into this reference file
- ✅ Moved all datasets to proper locations
- ✅ Created configuration files
- ✅ Implemented core modules (config, logger, landmark_extractor)
- ✅ Set up Docker infrastructure
- 📝 Status: Ready for Development

### Next Update
- Will add: Dataset processing progress
- Will add: Model training results
- Will add: API endpoint documentation

---

**Keep this file updated as you progress through development!**

This is your single source of truth for the SignBridge project.
