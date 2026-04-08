# SignBridge Implementation Guide
## Technical Deep Dive & Step-by-Step Development Plan

---

## Table of Contents
1. [Development Environment Setup](#development-environment-setup)
2. [Data Collection & Preparation](#data-collection--preparation)
3. [ML Model Development](#ml-model-development)
4. [Backend Architecture](#backend-architecture)
5. [Frontend Development](#frontend-development)
6. [Integration & Testing](#integration--testing)
7. [Research Methodology](#research-methodology)

---

## Development Environment Setup

### Required Tools & Dependencies

#### Core Development
```bash
# Python environment
conda create -n signbridge python=3.10
conda activate signbridge

# Core ML libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow tensorflow-gpu  # For MediaPipe compatibility
pip install transformers datasets huggingface-hub
pip install mediapipe opencv-python pillow

# Data processing
pip install numpy pandas scipy scikit-learn
pip install albumentations imgaug  # Data augmentation
pip install pytorchvideo  # Video processing

# Visualization & monitoring
pip install matplotlib seaborn plotly
pip install tensorboard wandb  # Experiment tracking
pip install grad-cam pytorch-lightning

# Backend
pip install fastapi uvicorn pydantic
pip install sqlalchemy psycopg2-binary
pip install redis celery
pip install python-multipart python-jose passlib

# Testing
pip install pytest pytest-cov pytest-asyncio
pip install locust  # Load testing
```

#### Frontend Setup
```bash
# Node.js and React
npm install -g create-react-app
npx create-react-app signbridge-frontend
cd signbridge-frontend

# Core dependencies
npm install @tensorflow/tfjs @mediapipe/holistic
npm install three @react-three/fiber @react-three/drei  # 3D avatar
npm install axios react-router-dom
npm install @mui/material @emotion/react @emotion/styled  # UI components
npm install react-webcam webrtc-adapter

# Development tools
npm install --save-dev @types/react @types/node
npm install --save-dev eslint prettier
```

#### Infrastructure
```bash
# Docker setup
docker --version  # Ensure Docker is installed
docker-compose --version

# Cloud CLI (choose based on provider)
pip install awscli boto3  # AWS
pip install google-cloud-storage  # GCP
```

### Project Structure

```
signbridge/
├── data/
│   ├── raw/                    # Raw video files
│   ├── processed/              # Preprocessed features
│   ├── annotations/            # Sign gloss annotations
│   └── datasets/               # Train/val/test splits
├── models/
│   ├── sign_recognition/       # Sign → Text models
│   │   ├── spatial_encoder.py
│   │   ├── temporal_encoder.py
│   │   ├── seq2seq_translator.py
│   │   └── pretrained/         # Checkpoint storage
│   ├── sign_generation/        # Text → Sign models
│   │   ├── nlp_processor.py
│   │   ├── motion_planner.py
│   │   └── avatar_animator.py
│   └── shared/
│       ├── feature_extractor.py
│       └── utils.py
├── backend/
│   ├── api/
│   │   ├── main.py            # FastAPI app
│   │   ├── routes/
│   │   ├── schemas/
│   │   └── dependencies.py
│   ├── services/
│   │   ├── translation_service.py
│   │   ├── learning_engine.py
│   │   └── user_service.py
│   ├── database/
│   │   ├── models.py
│   │   └── crud.py
│   └── ml_serving/
│       ├── model_loader.py
│       └── inference_pipeline.py
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── TranslationView.jsx
│   │   │   ├── LearningMode.jsx
│   │   │   ├── AvatarRenderer.jsx
│   │   │   └── VideoCapture.jsx
│   │   ├── services/
│   │   │   ├── api.js
│   │   │   ├── mediapipe.js
│   │   │   └── tfjs-inference.js
│   │   ├── utils/
│   │   └── App.jsx
│   └── package.json
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
├── scripts/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── deploy.sh
├── tests/
│   ├── test_models.py
│   ├── test_api.py
│   └── test_frontend.py
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── docs/
│   ├── API.md
│   ├── MODEL_ARCHITECTURE.md
│   └── USER_GUIDE.md
├── requirements.txt
├── setup.py
└── README.md
```

---

## Data Collection & Preparation

### Step 1: Dataset Acquisition

#### Public Datasets to Start With

**1. WLASL (Word-Level American Sign Language)**
```python
# Download script
import requests
import os
from pathlib import Path

def download_wlasl():
    base_url = "https://github.com/dxli94/WLASL/raw/master/"
    
    # Download annotations
    annotations_url = base_url + "data/WLASL_v0.3.json"
    response = requests.get(annotations_url)
    
    with open('data/raw/wlasl_annotations.json', 'wb') as f:
        f.write(response.content)
    
    # Download videos using annotations
    # Note: Actual implementation requires parsing JSON and downloading from YouTube
    print("WLASL annotations downloaded. Video download script needed.")

# Dataset stats: 2,000 words, 21,083 videos, 119 signers
```

**2. How2Sign (Sentence-Level ASL)**
```bash
# Register at https://how2sign.github.io/ and download
# Dataset: 16,000 sentence pairs with RGB video and depth

# Expected structure after download:
# data/raw/how2sign/
#   ├── train/
#   ├── val/
#   └── test/
```

**3. MS-ASL (Microsoft ASL Dataset)**
```python
# Available at: https://www.microsoft.com/en-us/research/project/ms-asl/
# 1,000 classes, 25,513 videos
# Download requires Microsoft account
```

**4. INCLUDE (Indian Sign Language)**
```bash
# Currently limited availability - 263 signs, 4,287 videos
# May require direct contact with researchers at IIT Delhi
# Alternative: Collect your own ISL data (see below)
```

#### Creating Your Own Dataset

**Recording Protocol**:
```python
# recording_setup.py
import cv2
import mediapipe as mp
import json
from datetime import datetime

class SignRecorder:
    def __init__(self, output_dir='data/raw/custom'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def record_sign(self, sign_label, signer_id, session_id):
        """Record a single sign with metadata"""
        cap = cv2.VideoCapture(0)
        
        # Set resolution (1280x720 for better hand detection)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"{sign_label}_{signer_id}_{session_id}_{timestamp}.mp4"
        video_path = self.output_dir / video_filename
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (1280, 720))
        
        frames = []
        landmarks_sequence = []
        
        print(f"Recording '{sign_label}'. Press 'r' to start, 's' to stop, 'q' to quit.")
        
        recording = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process with MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image_rgb)
            
            # Display frame
            display_frame = frame.copy()
            
            if recording:
                cv2.putText(display_frame, "RECORDING", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Save frame and landmarks
                out.write(frame)
                frames.append(frame)
                
                # Extract landmarks
                landmark_frame = self._extract_landmarks(results)
                landmarks_sequence.append(landmark_frame)
            
            cv2.imshow('Sign Recording', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                recording = True
                print("Started recording...")
            elif key == ord('s'):
                recording = False
                print(f"Stopped recording. Saved {len(frames)} frames.")
                break
            elif key == ord('q'):
                print("Quit without saving.")
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                return None
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Save metadata
        metadata = {
            'sign_label': sign_label,
            'signer_id': signer_id,
            'session_id': session_id,
            'timestamp': timestamp,
            'num_frames': len(frames),
            'fps': 30,
            'resolution': [1280, 720],
            'video_path': str(video_path)
        }
        
        metadata_path = video_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save landmarks
        landmarks_path = video_path.with_suffix('.npy')
        import numpy as np
        np.save(landmarks_path, np.array(landmarks_sequence))
        
        print(f"Saved: {video_path}")
        return metadata
    
    def _extract_landmarks(self, results):
        """Extract all landmarks from MediaPipe results"""
        landmarks = {}
        
        # Pose landmarks (33 points)
        if results.pose_landmarks:
            landmarks['pose'] = [
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in results.pose_landmarks.landmark
            ]
        
        # Left hand (21 points)
        if results.left_hand_landmarks:
            landmarks['left_hand'] = [
                [lm.x, lm.y, lm.z]
                for lm in results.left_hand_landmarks.landmark
            ]
        
        # Right hand (21 points)
        if results.right_hand_landmarks:
            landmarks['right_hand'] = [
                [lm.x, lm.y, lm.z]
                for lm in results.right_hand_landmarks.landmark
            ]
        
        # Face landmarks (468 points - focus on key points for expressions)
        if results.face_landmarks:
            # Extract subset for efficiency (eyebrows, eyes, mouth)
            key_indices = list(range(0, 17)) + list(range(33, 133)) + list(range(362, 400))
            landmarks['face'] = [
                [results.face_landmarks.landmark[i].x,
                 results.face_landmarks.landmark[i].y,
                 results.face_landmarks.landmark[i].z]
                for i in key_indices
            ]
        
        return landmarks

# Usage
if __name__ == "__main__":
    recorder = SignRecorder()
    
    # Record multiple signs
    signs_to_record = ['HELLO', 'THANK_YOU', 'PLEASE', 'SORRY', 'YES', 'NO']
    
    for sign in signs_to_record:
        recorder.record_sign(
            sign_label=sign,
            signer_id='signer_001',
            session_id='session_001'
        )
```

**Data Collection Guidelines**:
1. **Diversity**:
   - At least 10 different signers per sign
   - Varying ages, genders, skin tones, body types
   - Different backgrounds and lighting conditions
   
2. **Consistency**:
   - Same camera distance (arms fully extended should be visible)
   - 720p or 1080p resolution
   - 30 FPS minimum
   
3. **Annotation**:
   - Sign gloss labels
   - Start/end frame timestamps
   - Signer metadata (anonymized ID, handedness, fluency level)

### Step 2: Data Preprocessing

#### Video Preprocessing Pipeline

```python
# data_preprocessing.py
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
from tqdm import tqdm
import albumentations as A

class SignLanguagePreprocessor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # MediaPipe setup
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Augmentation pipeline (for training data)
        self.augmentation = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.MotionBlur(p=0.2),
            A.RandomGamma(p=0.3),
        ])
    
    def process_video(self, video_path, augment=False):
        """
        Process a video and extract landmarks
        Returns: dict with landmarks sequence and metadata
        """
        cap = cv2.VideoCapture(str(video_path))
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        landmarks_sequence = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Optional augmentation
            if augment:
                frame = self.augmentation(image=frame)['image']
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.holistic.process(image_rgb)
            
            # Extract and normalize landmarks
            frame_landmarks = self._extract_and_normalize_landmarks(results)
            landmarks_sequence.append(frame_landmarks)
            
            frame_idx += 1
        
        cap.release()
        
        return {
            'landmarks': np.array(landmarks_sequence),
            'fps': fps,
            'num_frames': frame_count,
            'video_path': str(video_path)
        }
    
    def _extract_and_normalize_landmarks(self, results):
        """
        Extract landmarks and normalize them
        Returns fixed-size feature vector
        """
        features = []
        
        # Pose landmarks (33 points × 4 features = 132)
        if results.pose_landmarks:
            pose_features = []
            for lm in results.pose_landmarks.landmark:
                pose_features.extend([lm.x, lm.y, lm.z, lm.visibility])
            features.extend(pose_features)
        else:
            features.extend([0.0] * 132)  # Padding if not detected
        
        # Left hand (21 points × 3 features = 63)
        if results.left_hand_landmarks:
            left_hand_features = []
            for lm in results.left_hand_landmarks.landmark:
                left_hand_features.extend([lm.x, lm.y, lm.z])
            features.extend(left_hand_features)
        else:
            features.extend([0.0] * 63)
        
        # Right hand (21 points × 3 features = 63)
        if results.right_hand_landmarks:
            right_hand_features = []
            for lm in results.right_hand_landmarks.landmark:
                right_hand_features.extend([lm.x, lm.y, lm.z])
            features.extend(right_hand_features)
        else:
            features.extend([0.0] * 63)
        
        # Face landmarks (reduced to key points: 50 × 3 = 150)
        if results.face_landmarks:
            # Select key facial points for expressions
            key_face_indices = [
                # Eyebrows
                70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
                # Eyes
                33, 133, 160, 159, 158, 157, 173, 263, 362, 385,
                # Mouth
                61, 291, 0, 17, 269, 270, 409, 291, 375, 321,
                # Nose
                1, 2, 98, 327,
                # Jawline
                172, 136, 150, 176, 152, 400, 379, 365, 397, 288
            ]
            
            face_features = []
            for idx in key_face_indices:
                lm = results.face_landmarks.landmark[idx]
                face_features.extend([lm.x, lm.y, lm.z])
            features.extend(face_features)
        else:
            features.extend([0.0] * 150)
        
        # Total: 132 + 63 + 63 + 150 = 408 features per frame
        return np.array(features, dtype=np.float32)
    
    def process_dataset(self, annotation_file, split='train'):
        """
        Process entire dataset based on annotations
        """
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        processed_data = []
        
        for item in tqdm(annotations, desc=f"Processing {split}"):
            video_path = self.input_dir / item['video_path']
            
            if not video_path.exists():
                print(f"Warning: {video_path} not found")
                continue
            
            # Process video
            augment = (split == 'train')  # Only augment training data
            processed = self.process_video(video_path, augment=augment)
            
            # Add annotation info
            processed['label'] = item['sign_label']
            processed['gloss'] = item.get('gloss', item['sign_label'])
            processed['signer_id'] = item.get('signer_id', 'unknown')
            
            processed_data.append(processed)
        
        # Save processed data
        output_path = self.output_dir / f'{split}_processed.pkl'
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        print(f"Processed {len(processed_data)} videos for {split}")
        return processed_data

# Usage
if __name__ == "__main__":
    preprocessor = SignLanguagePreprocessor(
        input_dir='data/raw',
        output_dir='data/processed'
    )
    
    # Process each split
    for split in ['train', 'val', 'test']:
        annotation_file = f'data/annotations/{split}_annotations.json'
        preprocessor.process_dataset(annotation_file, split=split)
```

#### Feature Engineering

```python
# feature_engineering.py
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter

class SignFeatureEngineer:
    """
    Extract higher-level features from raw landmarks
    """
    
    @staticmethod
    def compute_hand_shape_features(hand_landmarks):
        """
        Compute hand shape descriptors
        - Finger extensions
        - Palm orientation
        - Hand openness
        """
        if hand_landmarks is None or len(hand_landmarks) == 0:
            return np.zeros(10)
        
        # Reshape to (21, 3) if flattened
        if hand_landmarks.shape == (63,):
            hand_landmarks = hand_landmarks.reshape(21, 3)
        
        features = []
        
        # Finger tip distances from palm center (wrist)
        wrist = hand_landmarks[0]
        fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        
        for tip_idx in fingertips:
            dist = euclidean(hand_landmarks[tip_idx], wrist)
            features.append(dist)
        
        # Hand openness (average distance between adjacent fingers)
        openness = 0
        for i in range(len(fingertips) - 1):
            openness += euclidean(hand_landmarks[fingertips[i]], 
                                hand_landmarks[fingertips[i+1]])
        features.append(openness / 4)
        
        # Palm orientation (using normal vector)
        # Define palm plane with wrist, index MCP, pinky MCP
        p1 = hand_landmarks[0]   # Wrist
        p2 = hand_landmarks[5]   # Index MCP
        p3 = hand_landmarks[17]  # Pinky MCP
        
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        
        features.extend(normal)  # 3 features
        
        return np.array(features)
    
    @staticmethod
    def compute_motion_features(landmarks_sequence, window=5):
        """
        Compute temporal motion features
        - Velocity (first derivative)
        - Acceleration (second derivative)
        """
        if len(landmarks_sequence) < window:
            return landmarks_sequence
        
        # Smooth with Savitzky-Golay filter
        smoothed = savgol_filter(landmarks_sequence, window, 3, axis=0)
        
        # Velocity (first derivative)
        velocity = np.gradient(smoothed, axis=0)
        
        # Acceleration (second derivative)
        acceleration = np.gradient(velocity, axis=0)
        
        # Concatenate: [position, velocity, acceleration]
        enhanced = np.concatenate([
            smoothed,
            velocity,
            acceleration
        ], axis=-1)
        
        return enhanced
    
    @staticmethod
    def compute_spatial_features(landmarks_frame):
        """
        Compute spatial relationships between body parts
        - Hand-to-face distance
        - Hand-to-hand distance
        - Hand position relative to body center
        """
        # Extract components (assuming 408-dim vector)
        pose = landmarks_frame[:132].reshape(33, 4)[:, :3]  # Only x,y,z
        left_hand = landmarks_frame[132:195].reshape(21, 3)
        right_hand = landmarks_frame[195:258].reshape(21, 3)
        face = landmarks_frame[258:408].reshape(50, 3)
        
        features = []
        
        # Hand-to-face distance (use nose as face reference)
        nose = pose[0]  # Nose landmark from pose
        
        if np.any(left_hand != 0):
            left_wrist = left_hand[0]
            left_to_face = euclidean(left_wrist, nose)
            features.append(left_to_face)
        else:
            features.append(0.0)
        
        if np.any(right_hand != 0):
            right_wrist = right_hand[0]
            right_to_face = euclidean(right_wrist, nose)
            features.append(right_to_face)
        else:
            features.append(0.0)
        
        # Hand-to-hand distance
        if np.any(left_hand != 0) and np.any(right_hand != 0):
            hand_distance = euclidean(left_hand[0], right_hand[0])
            features.append(hand_distance)
        else:
            features.append(0.0)
        
        # Hand height relative to shoulders
        left_shoulder = pose[11]
        right_shoulder = pose[12]
        shoulder_midpoint = (left_shoulder + right_shoulder) / 2
        
        if np.any(left_hand != 0):
            left_height = left_hand[0][1] - shoulder_midpoint[1]
            features.append(left_height)
        else:
            features.append(0.0)
        
        if np.any(right_hand != 0):
            right_height = right_hand[0][1] - shoulder_midpoint[1]
            features.append(right_height)
        else:
            features.append(0.0)
        
        return np.array(features)
```

### Step 3: Data Annotation & Augmentation

```python
# annotation_tool.py (simplified labeling interface)
import cv2
import json
from pathlib import Path

class AnnotationTool:
    def __init__(self, video_dir, output_file):
        self.video_dir = Path(video_dir)
        self.output_file = output_file
        self.annotations = []
        
    def annotate_video(self, video_path):
        """
        Simple annotation interface for sign gloss
        """
        cap = cv2.VideoCapture(str(video_path))
        
        print(f"\nAnnotating: {video_path.name}")
        print("Controls:")
        print("  SPACE - Pause/Play")
        print("  'a' - Add annotation")
        print("  'q' - Quit")
        
        paused = False
        frame_idx = 0
        
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
            
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Annotation Tool', frame)
            
            key = cv2.waitKey(30 if not paused else 0) & 0xFF
            
            if key == ord(' '):
                paused = not paused
            elif key == ord('a'):
                gloss = input("\nEnter sign gloss: ")
                start_frame = int(input("Start frame: "))
                end_frame = int(input("End frame: "))
                
                annotation = {
                    'video_path': str(video_path),
                    'sign_label': gloss,
                    'gloss': gloss,
                    'start_frame': start_frame,
                    'end_frame': end_frame
                }
                self.annotations.append(annotation)
                print(f"Added annotation: {annotation}")
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def save_annotations(self):
        """Save all annotations to JSON"""
        with open(self.output_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        print(f"Saved {len(self.annotations)} annotations to {self.output_file}")

# For larger-scale annotation, consider Label Studio or CVAT
# Label Studio: https://labelstud.io/
# CVAT: https://github.com/opencv/cvat
```

---

## ML Model Development

### Architecture 1: Sign Recognition (Sign → Text)

#### Model Definition

```python
# models/sign_recognition/spatial_temporal_model.py
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class SpatialEncoder(nn.Module):
    """
    Encodes spatial features from a single frame
    """
    def __init__(self, input_dim=408, hidden_dim=256):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(0.3)
        
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        batch, seq_len, input_dim = x.shape
        
        # Reshape for batch norm
        x = x.view(batch * seq_len, input_dim)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        
        # Reshape back
        x = x.view(batch, seq_len, -1)
        
        return x


class TemporalEncoder(nn.Module):
    """
    Encodes temporal dynamics using Transformer
    """
    def __init__(self, input_dim=256, num_layers=4, nhead=8, dim_feedforward=1024):
        super().__init__()
        
        encoder_layers = TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(input_dim)
    
    def forward(self, x, src_key_padding_mask=None):
        # x shape: (batch, seq_len, input_dim)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class SignRecognitionModel(nn.Module):
    """
    Complete model: Spatial + Temporal encoding + Classification
    """
    def __init__(self, 
                 input_dim=408,
                 spatial_dim=256,
                 temporal_dim=256,
                 num_classes=1000,
                 num_transformer_layers=4):
        super().__init__()
        
        self.spatial_encoder = SpatialEncoder(input_dim, spatial_dim)
        self.temporal_encoder = TemporalEncoder(
            spatial_dim, 
            num_layers=num_transformer_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(temporal_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x, lengths=None):
        # x shape: (batch, seq_len, input_dim)
        
        # Spatial encoding
        spatial_features = self.spatial_encoder(x)
        
        # Create padding mask if lengths provided
        if lengths is not None:
            batch_size, max_len = x.size(0), x.size(1)
            mask = torch.arange(max_len).expand(batch_size, max_len).to(x.device)
            mask = mask >= lengths.unsqueeze(1)
        else:
            mask = None
        
        # Temporal encoding
        temporal_features = self.temporal_encoder(spatial_features, mask)
        
        # Global pooling (mean over time, ignoring padding)
        if mask is not None:
            temporal_features = temporal_features.masked_fill(mask.unsqueeze(-1), 0)
            pooled = temporal_features.sum(dim=1) / lengths.unsqueeze(1).float()
        else:
            pooled = temporal_features.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits


# For sequence-to-sequence (sign gloss → text translation)
class Seq2SeqTranslator(nn.Module):
    """
    Translates sign gloss sequence to natural language
    """
    def __init__(self, 
                 vocab_size_src,
                 vocab_size_tgt,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6):
        super().__init__()
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        
        self.src_embedding = nn.Embedding(vocab_size_src, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size_tgt, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size_tgt)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embedding + positional encoding
        src = self.pos_encoder(self.src_embedding(src))
        tgt = self.pos_encoder(self.tgt_embedding(tgt))
        
        # Transformer
        output = self.transformer(
            src, tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
        
        # Project to vocabulary
        output = self.fc_out(output)
        
        return output
```

#### Training Script

```python
# scripts/train_sign_recognition.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pickle

class SignLanguageDataset(Dataset):
    def __init__(self, data_file, max_seq_len=150):
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        
        self.max_seq_len = max_seq_len
        
        # Build label mapping
        unique_labels = set(item['label'] for item in self.data)
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get landmarks
        landmarks = item['landmarks']  # Shape: (num_frames, 408)
        
        # Truncate or pad
        if len(landmarks) > self.max_seq_len:
            landmarks = landmarks[:self.max_seq_len]
        else:
            pad_len = self.max_seq_len - len(landmarks)
            landmarks = np.pad(landmarks, ((0, pad_len), (0, 0)), 'constant')
        
        # Get label
        label = self.label_to_idx[item['label']]
        
        # Actual sequence length (for masking)
        seq_len = min(len(item['landmarks']), self.max_seq_len)
        
        return {
            'landmarks': torch.FloatTensor(landmarks),
            'label': torch.LongTensor([label]),
            'seq_len': torch.LongTensor([seq_len])
        }


class SignRecognitionLightning(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-4):
        super().__init__()
        
        self.model = SignRecognitionModel(
            input_dim=408,
            num_classes=num_classes
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        
        self.save_hyperparameters()
    
    def forward(self, x, lengths):
        return self.model(x, lengths)
    
    def training_step(self, batch, batch_idx):
        landmarks = batch['landmarks']
        labels = batch['label'].squeeze()
        lengths = batch['seq_len'].squeeze()
        
        logits = self(landmarks, lengths)
        loss = self.criterion(logits, labels)
        
        # Accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        landmarks = batch['landmarks']
        labels = batch['label'].squeeze()
        lengths = batch['seq_len'].squeeze()
        
        logits = self(landmarks, lengths)
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


# Training script
def train_model():
    # Load data
    train_dataset = SignLanguageDataset('data/processed/train_processed.pkl')
    val_dataset = SignLanguageDataset('data/processed/val_processed.pkl')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # Model
    num_classes = len(train_dataset.label_to_idx)
    model = SignRecognitionLightning(num_classes=num_classes)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/checkpoints',
        filename='sign-recognition-{epoch:02d}-{val_acc:.2f}',
        monitor='val_acc',
        mode='max',
        save_top_k=3
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='gpu',
        devices=1,
        precision=16  # Mixed precision training
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    train_model()
```

This implementation guide provides:

1. **Complete environment setup** with all dependencies
2. **Data collection** strategies for both public datasets and custom recording
3. **Preprocessing pipeline** with MediaPipe integration
4. **Feature engineering** for improved performance
5. **ML architecture** with spatial-temporal modeling
6. **Training infrastructure** using PyTorch Lightning

Would you like me to continue with:
- Backend API implementation (FastAPI)
- Frontend development (React + Three.js avatar)
- Text → Sign generation pipeline
- Research methodology and evaluation metrics
- Deployment strategy

Let me know which sections to expand next!
