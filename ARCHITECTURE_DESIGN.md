# 🏗️ SignBridge Architecture Design

## Unified Architecture Based on Best Practices

### **Core Principles**
1. **Modular Design** - Clean separation of concerns
2. **Extensible Framework** - Easy to add new features
3. **Real-Time Performance** - Optimized for live inference
4. **Production Ready** - Scalable, maintainable code

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Real-Time    │  │ Learning    │  │ Phrase       │     │
│  │ Detection    │  │ Mode        │  │ Dictionary   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP/WebSocket
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend API (FastAPI)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Prediction   │  │ Learning     │  │ Data         │     │
│  │ Service     │  │ Service      │  │ Collection   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  ML Pipeline (Python)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Feature      │  │ Model        │  │ Inference    │     │
│  │ Extraction   │  │ Training     │  │ Engine       │     │
│  │ (MediaPipe)  │  │ (LSTM/MLP)   │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
signbridge/
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── core/                          # Core ML components
│   ├── __init__.py
│   ├── feature_extractor.py       # MediaPipe integration
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py          # LSTM for sequences
│   │   ├── mlp_model.py           # MLP for fast inference
│   │   └── base_model.py          # Base model interface
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collector.py           # Interactive data collection
│   │   ├── preprocessor.py        # Data preprocessing
│   │   └── dataset.py             # Dataset management
│   └── training/
│       ├── __init__.py
│       ├── trainer.py             # Training pipeline
│       └── callbacks.py            # Training callbacks
│
├── backend/                       # FastAPI backend
│   ├── __init__.py
│   ├── main.py                    # FastAPI app
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── prediction.py      # Prediction endpoints
│   │   │   ├── learning.py        # Learning mode endpoints
│   │   │   └── data.py            # Data collection endpoints
│   │   └── schemas.py             # Pydantic schemas
│   ├── services/
│   │   ├── __init__.py
│   │   ├── inference.py           # Inference service
│   │   └── model_loader.py        # Model loading service
│   └── config.py                  # Configuration
│
├── frontend/                      # React frontend
│   ├── package.json
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── VideoCapture.jsx
│   │   │   ├── PredictionDisplay.jsx
│   │   │   ├── LearningMode.jsx
│   │   │   └── PhraseDictionary.jsx
│   │   ├── services/
│   │   │   └── api.js
│   │   └── utils/
│   │       └── mediapipe.js
│   └── public/
│
├── data/                          # Data storage
│   ├── raw/                       # Raw videos/images
│   ├── processed/                # Processed landmarks
│   │   ├── landmarks/            # CSV/NPY files
│   │   └── sequences/            # Sequence data
│   ├── models/                   # Trained models
│   │   ├── lstm/
│   │   └── mlp/
│   └── annotations/              # Labels and metadata
│
├── scripts/                       # Utility scripts
│   ├── train.py                  # Training script
│   ├── collect_data.py           # Data collection script
│   ├── preprocess.py             # Preprocessing script
│   └── evaluate.py               # Evaluation script
│
├── tests/                         # Tests
│   ├── test_models.py
│   ├── test_api.py
│   └── test_features.py
│
└── docs/                          # Documentation
    ├── API.md
    ├── ARCHITECTURE.md
    └── USER_GUIDE.md
```

---

## Core Components Design

### 1. Feature Extractor (`core/feature_extractor.py`)

```python
class FeatureExtractor:
    """MediaPipe-based feature extraction"""
    
    def __init__(self, mode='holistic'):
        # MediaPipe Holistic or Hands
        self.mode = mode
        self.mp_holistic = mp.solutions.holistic
        self.mp_hands = mp.solutions.hands
    
    def extract_landmarks(self, image):
        """Extract landmarks from image"""
        # Returns: pose, face, left_hand, right_hand landmarks
        pass
    
    def extract_sequence(self, video_path):
        """Extract landmarks from video sequence"""
        pass
    
    def normalize_landmarks(self, landmarks):
        """Normalize landmarks to fixed-size vector"""
        # Holistic: 1662 dims
        # Hands only: 126 dims (2 hands)
        pass
```

### 2. LSTM Model (`core/models/lstm_model.py`)

```python
class SignLSTMModel(nn.Module):
    """LSTM model for sign recognition"""
    
    def __init__(self, input_dim=1662, hidden_dim=128, num_classes=30):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True, return_sequences=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim*2, batch_first=True, return_sequences=True)
        self.lstm3 = nn.LSTM(hidden_dim*2, hidden_dim, batch_first=True, return_sequences=False)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### 3. MLP Model (`core/models/mlp_model.py`)

```python
class SignMLPModel(nn.Module):
    """MLP model for fast inference"""
    
    def __init__(self, input_dim=126, hidden_dims=[256, 128, 64], num_classes=30):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
```

### 4. Data Collector (`core/data/collector.py`)

```python
class DataCollector:
    """Interactive data collection"""
    
    def __init__(self, output_dir='data/raw'):
        self.output_dir = Path(output_dir)
        self.feature_extractor = FeatureExtractor()
    
    def collect_sequence(self, sign_label, num_sequences=30, seq_length=30):
        """Collect sequences for a sign"""
        # Interactive collection with webcam
        # Save landmarks to CSV/NPY
        pass
    
    def collect_batch(self, signs, sequences_per_sign=30):
        """Collect data for multiple signs"""
        pass
```

### 5. Training Pipeline (`core/training/trainer.py`)

```python
class SignTrainer:
    """Training pipeline"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train(self, train_loader, val_loader):
        """Train model"""
        # Training loop with callbacks
        pass
    
    def evaluate(self, test_loader):
        """Evaluate model"""
        pass
```

### 6. Inference Service (`backend/services/inference.py`)

```python
class InferenceService:
    """Real-time inference service"""
    
    def __init__(self, model_path, model_type='lstm'):
        self.model = self.load_model(model_path, model_type)
        self.feature_extractor = FeatureExtractor()
        self.sequence_buffer = deque(maxlen=30)
    
    def predict(self, image):
        """Predict sign from image"""
        landmarks = self.feature_extractor.extract_landmarks(image)
        self.sequence_buffer.append(landmarks)
        
        if len(self.sequence_buffer) == 30:
            sequence = np.array(self.sequence_buffer)
            prediction = self.model.predict(sequence)
            return prediction
        return None
    
    def predict_sequence(self, sequence):
        """Predict from pre-extracted sequence"""
        pass
```

---

## API Design

### Endpoints

```
POST   /api/v1/predict          # Single frame prediction
POST   /api/v1/predict/sequence  # Sequence prediction
GET    /api/v1/health            # Health check
POST   /api/v1/learning/start     # Start learning session
POST   /api/v1/learning/feedback  # Submit feedback
GET    /api/v1/phrases           # Get phrase dictionary
POST   /api/v1/data/collect      # Submit collected data
```

### Request/Response Examples

```python
# POST /api/v1/predict
{
  "image": "base64_encoded_image",
  "model_type": "lstm"  # or "mlp"
}

# Response
{
  "prediction": "HELLO",
  "confidence": 0.95,
  "probabilities": {...}
}

# POST /api/v1/predict/sequence
{
  "sequence": [[...], [...], ...],  # 30 frames of landmarks
  "model_type": "lstm"
}

# POST /api/v1/learning/feedback
{
  "sign": "HELLO",
  "correct": true,
  "user_correction": null
}
```

---

## Data Flow

### Training Flow
```
1. Data Collection (Interactive)
   ↓
2. Feature Extraction (MediaPipe)
   ↓
3. Preprocessing (Normalization, Augmentation)
   ↓
4. Dataset Creation (Train/Val/Test splits)
   ↓
5. Model Training (LSTM/MLP)
   ↓
6. Model Evaluation
   ↓
7. Model Deployment
```

### Inference Flow
```
1. Webcam Frame
   ↓
2. Feature Extraction (MediaPipe)
   ↓
3. Sequence Buffer (30 frames)
   ↓
4. Model Inference (LSTM/MLP)
   ↓
5. Post-processing (Smoothing, Confidence)
   ↓
6. Prediction Display
```

---

## Technology Stack

### Backend
- **FastAPI** - Modern async web framework
- **PyTorch** - Deep learning framework
- **MediaPipe** - Feature extraction
- **NumPy/Pandas** - Data processing
- **Uvicorn** - ASGI server

### Frontend
- **React** - UI framework
- **TensorFlow.js** - Browser ML (optional)
- **MediaPipe** (JS) - Client-side features (optional)
- **Axios** - HTTP client
- **WebRTC** - Camera access

### Data Storage
- **CSV/NPY** - Landmark data
- **HDF5** - Large datasets
- **MongoDB** - Phrase dictionary (future)
- **PostgreSQL** - User data (future)

---

## Performance Targets

- **Latency**: < 100ms per prediction
- **Accuracy**: > 95% on trained signs
- **Throughput**: 30 FPS real-time inference
- **Model Size**: < 50MB for LSTM, < 10MB for MLP

---

## Next Steps

1. ✅ Create project structure
2. ✅ Implement feature extractor
3. ✅ Implement LSTM model
4. ✅ Implement MLP model
5. ✅ Implement data collector
6. ✅ Implement training pipeline
7. ✅ Build FastAPI backend
8. ✅ Build React frontend
9. ✅ Integration and testing

---

**Ready to implement!** 🚀


