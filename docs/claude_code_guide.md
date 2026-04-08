# SignBridge: Claude Code Implementation Guide
## Complete Step-by-Step Guide for Building with Claude Code

---

## Table of Contents
1. [Introduction to Claude Code](#introduction-to-claude-code)
2. [Project Setup Phase](#project-setup-phase)
3. [Phase 1: Data Pipeline (Weeks 1-2)](#phase-1-data-pipeline)
4. [Phase 2: ML Models (Weeks 3-4)](#phase-2-ml-models)
5. [Phase 3: Backend API (Weeks 5-6)](#phase-3-backend-api)
6. [Phase 4: Frontend (Weeks 7-8)](#phase-4-frontend)
7. [Phase 5: Integration & Testing (Weeks 9-10)](#phase-5-integration--testing)
8. [Phase 6: Deployment (Weeks 11-12)](#phase-6-deployment)
9. [Prompt Engineering Tips](#prompt-engineering-tips)
10. [Troubleshooting Guide](#troubleshooting-guide)

---

## Introduction to Claude Code

### What is Claude Code?
Claude Code is a command-line tool that allows you to delegate coding tasks to Claude directly from your terminal. It's designed for agentic coding where Claude can:
- Read and write files across your entire project
- Execute commands and tests
- Iterate on code based on results
- Work autonomously with your guidance

### Setting Up Claude Code

```bash
# Install Claude Code (follow latest installation from Anthropic docs)
# Check installation
claude-code --version

# Authenticate
claude-code auth login

# Initialize your project
mkdir signbridge
cd signbridge
claude-code init
```

### Project Configuration

Create a `.claude-code/config.yaml`:
```yaml
project:
  name: SignBridge
  description: Bidirectional sign language translation and learning platform
  
preferences:
  model: claude-sonnet-4-5-20250929
  max_iterations: 50
  
directories:
  source: ./
  tests: ./tests
  docs: ./docs
  
languages:
  - python
  - javascript
  - typescript
  
frameworks:
  - pytorch
  - react
  - fastapi
```

---

## Project Setup Phase

### Step 1: Initialize Project Structure

**Claude Code Prompt:**
```
Create the complete project structure for SignBridge with the following requirements:

1. Root directory structure:
   - data/ (with subdirs: raw, processed, annotations, datasets)
   - models/ (with subdirs: sign_recognition, sign_generation, shared, checkpoints)
   - backend/ (with subdirs: api, services, database, ml_serving)
   - frontend/ (React app structure)
   - notebooks/
   - scripts/
   - tests/
   - docker/
   - docs/

2. Create initial configuration files:
   - requirements.txt with all Python dependencies (PyTorch, FastAPI, MediaPipe, etc.)
   - package.json for frontend (React, Three.js, TensorFlow.js)
   - .gitignore for Python and Node.js
   - docker-compose.yml for development environment
   - README.md with project overview

3. Create placeholder __init__.py files for all Python modules

4. Set up basic logging configuration in backend/utils/logger.py

5. Create environment variable templates (.env.example)

Execute this and show me the final directory tree.
```

**Expected Output:**
Claude Code will create the entire structure and show you a tree view. Review it and ask for modifications if needed.

**Follow-up Prompt if needed:**
```
Adjust the structure to add:
- backend/middleware/ for authentication and CORS
- models/evaluation/ for metrics and benchmarking
- frontend/src/hooks/ for React custom hooks
- scripts/deployment/ for deployment scripts
```

---

## Phase 1: Data Pipeline (Weeks 1-2)

### Task 1.1: Data Collection Setup

**Claude Code Prompt:**
```
Implement a complete data collection and preprocessing pipeline with these specifications:

CONTEXT:
We're building a sign language recognition system. We need to:
1. Process video files to extract pose/hand/face landmarks using MediaPipe
2. Handle multiple sign languages (ASL, ISL, BSL, JSL, LSF)
3. Support both public datasets and custom recordings

REQUIREMENTS:

1. Create scripts/data_collection/video_recorder.py:
   - Class: SignVideoRecorder
   - Use OpenCV to capture video from webcam at 720p, 30fps
   - Integrate MediaPipe Holistic for real-time landmark visualization
   - Allow user to start/stop recording with keyboard shortcuts
   - Save video files with metadata (JSON) including:
     * sign_label
     * signer_id
     * timestamp
     * num_frames
     * fps
     * resolution
   - Save extracted landmarks as .npy files alongside videos

2. Create scripts/data_preprocessing/landmark_extractor.py:
   - Class: LandmarkExtractor
   - Extract 3D landmarks from video files:
     * Pose: 33 points × 4 features (x, y, z, visibility) = 132
     * Left hand: 21 points × 3 features = 63
     * Right hand: 21 points × 3 features = 63
     * Face (key points): 50 points × 3 features = 150
     * Total: 408 features per frame
   - Handle missing landmarks (padding with zeros)
   - Normalize coordinates relative to frame dimensions
   - Support batch processing of video directories
   - Include progress bars with tqdm

3. Create scripts/data_preprocessing/feature_engineering.py:
   - Class: SignFeatureEngineer
   - Implement feature extraction methods:
     * compute_hand_shape_features() - finger extensions, palm orientation
     * compute_motion_features() - velocity and acceleration using gradient
     * compute_spatial_features() - hand-face distance, hand-hand distance
   - Use Savitzky-Golay filter for smoothing
   - All methods should handle missing data gracefully

4. Create data augmentation in scripts/data_preprocessing/augmentation.py:
   - Use albumentations for image augmentation
   - Implement temporal augmentation (speed variation, frame dropping)
   - Maintain landmark consistency during augmentation

5. Create scripts/data_preprocessing/dataset_builder.py:
   - Process annotation JSON files
   - Split data into train/val/test (70/15/15)
   - Save processed datasets as pickled files
   - Generate dataset statistics report

6. Write comprehensive error handling for:
   - Missing video files
   - Corrupted videos
   - MediaPipe detection failures
   - Disk space issues

7. Include logging throughout with appropriate log levels

TEST REQUIREMENTS:
- Create tests/test_data_pipeline.py with unit tests for each component
- Test with sample video file
- Verify landmark extraction shape and data types
- Test augmentation preserves data integrity

Run the implementation and tests, show me results and any errors.
```

**What Claude Code Will Do:**
1. Create all the files with complete implementations
2. Run the code and show you any errors
3. Iterate to fix issues
4. Run tests and show results

**Follow-up Prompt for Debugging:**
```
The landmark extraction is failing for some videos. Debug by:
1. Adding more verbose logging to show which step fails
2. Adding try-catch around MediaPipe processing
3. Creating a script to identify problematic videos
4. Implement fallback strategies for partial landmark detection

Run the updated code on the test dataset and show me the success rate.
```

### Task 1.2: Dataset Download and Preparation

**Claude Code Prompt:**
```
Create dataset download and preparation scripts:

1. scripts/data_collection/download_wlasl.py:
   - Download WLASL annotations from GitHub
   - Parse JSON and create download list
   - Use youtube-dl or yt-dlp to download videos
   - Implement rate limiting and retry logic
   - Track download progress and failures
   - Create local annotation files in our format

2. scripts/data_collection/download_how2sign.py:
   - Script to help user register and download How2Sign
   - Instructions for manual download if needed
   - Processing script to convert to our format

3. scripts/data_collection/dataset_validator.py:
   - Validate downloaded datasets:
     * Check file integrity
     * Verify video can be read
     * Confirm annotation consistency
     * Check for duplicate entries
   - Generate validation report

4. Create data/README.md documenting:
   - Dataset sources and citations
   - Download instructions
   - Expected directory structure
   - Licensing information

CONSTRAINTS:
- Respect rate limits for downloads
- Include proper error handling and retries
- Make it resumable (don't re-download existing files)
- Support parallel downloads where appropriate

Execute and show me the download statistics for a small test batch (10 videos).
```

### Task 1.3: Data Analysis and Visualization

**Claude Code Prompt:**
```
Create data analysis and visualization tools in notebooks/data_exploration.ipynb:

1. Dataset Statistics:
   - Count videos per sign/gloss
   - Distribution of video lengths
   - Number of unique signers
   - Class imbalance analysis

2. Visualizations:
   - Histogram of sequence lengths
   - Bar chart of top 20 most common signs
   - Heatmap showing sign co-occurrence
   - Sample frame visualization with landmarks overlaid

3. Quality Analysis:
   - Detection success rates per video
   - Identify videos with poor landmark detection
   - Flag potential annotation errors

4. Create scripts/analysis/dataset_report.py to generate HTML report

Use pandas, matplotlib, seaborn, and plotly. Make visualizations interactive where possible.

Execute the notebook and save the report as data/dataset_report.html.
```

---

## Phase 2: ML Models (Weeks 3-4)

### Task 2.1: Sign Recognition Model Architecture

**Claude Code Prompt:**
```
Implement the complete sign recognition model architecture with these specifications:

CONTEXT:
We need a spatial-temporal model that processes landmark sequences to classify signs.
Architecture: SpatialEncoder → TemporalEncoder → Classifier

REQUIREMENTS:

1. Create models/sign_recognition/spatial_encoder.py:
   - Class: SpatialEncoder(nn.Module)
   - Input: (batch, seq_len, 408) landmark features
   - Architecture:
     * FC layer: 408 → 256, BatchNorm, ReLU, Dropout(0.3)
     * FC layer: 256 → 256, BatchNorm, ReLU, Dropout(0.3)
   - Output: (batch, seq_len, 256)
   - Handle batch processing properly for BatchNorm

2. Create models/sign_recognition/temporal_encoder.py:
   - Class: TemporalEncoder(nn.Module)
   - Input: (batch, seq_len, 256)
   - Architecture:
     * Positional encoding
     * Transformer encoder with 4 layers, 8 heads
     * Use nn.TransformerEncoder with batch_first=True
   - Support padding mask for variable-length sequences
   - Output: (batch, seq_len, 256)

3. Create models/sign_recognition/classifier.py:
   - Class: SignClassifier(nn.Module)
   - Input: (batch, 256) - pooled features
   - Architecture:
     * FC: 256 → 512, ReLU, Dropout(0.3)
     * FC: 512 → num_classes
   - Output: (batch, num_classes) logits

4. Create models/sign_recognition/model.py:
   - Class: SignRecognitionModel(nn.Module)
   - Combine all components:
     * spatial_encoder
     * temporal_encoder
     * classifier
   - Implement forward pass with:
     * Spatial encoding
     * Temporal encoding with masking
     * Mean pooling over time (masked)
     * Classification
   - Add model summary method

5. Create models/shared/positional_encoding.py:
   - Class: PositionalEncoding(nn.Module)
   - Sinusoidal positional encoding
   - Support max sequence length of 300 frames

6. Create models/sign_recognition/attention_visualizer.py:
   - Extract and visualize attention weights
   - Create attention heatmaps

TESTING:
- Create tests/test_models.py
- Test each component with sample tensors
- Verify shapes at each stage
- Test with variable-length sequences
- Check gradient flow
- Profile memory usage and speed

Run implementation, tests, and show me:
1. Model architecture summary
2. Test results
3. Sample forward pass output shapes
```

**Follow-up Prompts:**

**For Model Improvement:**
```
Enhance the model with these advanced features:

1. Add residual connections in SpatialEncoder
2. Implement multi-head attention pooling instead of mean pooling
3. Add learned positional embeddings as alternative to sinusoidal
4. Create model variants:
   - SignRecognitionModelLite (for mobile/edge)
   - SignRecognitionModelLarge (for maximum accuracy)

5. Add model export functionality:
   - Export to ONNX format
   - Export to TorchScript
   - Test exported models

Show me performance comparison between base and enhanced models.
```

### Task 2.2: Training Pipeline

**Claude Code Prompt:**
```
Create a complete training pipeline using PyTorch Lightning:

REQUIREMENTS:

1. Create models/sign_recognition/dataset.py:
   - Class: SignLanguageDataset(torch.utils.data.Dataset)
   - Load processed pickle files
   - Return dict with:
     * 'landmarks': (seq_len, 408)
     * 'label': int
     * 'seq_len': int
     * 'signer_id': str
   - Handle variable-length sequences
   - Implement __len__ and __getitem__

2. Create custom collate function for DataLoader:
   - Pad sequences to max length in batch
   - Create attention masks
   - Batch all metadata

3. Create models/sign_recognition/lightning_module.py:
   - Class: SignRecognitionLightning(pl.LightningModule)
   - Wrap SignRecognitionModel
   - Implement:
     * training_step: compute loss and accuracy
     * validation_step: compute metrics
     * test_step: comprehensive evaluation
     * configure_optimizers: AdamW + ReduceLROnPlateau
   - Log metrics to tensorboard
   - Add confusion matrix logging

4. Create scripts/train_model.py:
   - Load train/val datasets
   - Create DataLoaders with appropriate batch size
   - Initialize model
   - Set up callbacks:
     * ModelCheckpoint (save top 3 by val accuracy)
     * EarlyStopping (patience=10)
     * LearningRateMonitor
     * TQDMProgressBar
   - Create trainer with:
     * Mixed precision (16-bit)
     * Gradient clipping
     * GPU acceleration
   - Implement resumable training
   - Save training config and hyperparameters

5. Create models/evaluation/metrics.py:
   - Functions for:
     * Top-1 and Top-5 accuracy
     * Per-class accuracy
     * Confusion matrix
     * Precision, recall, F1-score
     * Per-signer performance analysis

6. Create scripts/evaluate_model.py:
   - Load best checkpoint
   - Run comprehensive evaluation on test set
   - Generate evaluation report with:
     * Overall metrics
     * Per-class performance
     * Confusion matrix visualization
     * Failure case analysis
   - Save report as HTML

TRAINING PARAMETERS:
- Batch size: 32
- Learning rate: 1e-4
- Max epochs: 100
- Optimizer: AdamW with weight decay 0.01
- Loss: CrossEntropyLoss with label smoothing 0.1

Run a short training test (5 epochs) on a small dataset subset and show me:
1. Training metrics
2. Validation metrics
3. Learning curve plot
4. Any warnings or errors
```

**For Hyperparameter Tuning:**
```
Implement hyperparameter search:

1. Create scripts/tune_hyperparameters.py:
   - Use Optuna for hyperparameter optimization
   - Search space:
     * learning_rate: [1e-5, 1e-3]
     * batch_size: [16, 32, 64]
     * hidden_dim: [128, 256, 512]
     * num_transformer_layers: [2, 4, 6]
     * dropout: [0.1, 0.3, 0.5]
   - Objective: maximize validation accuracy
   - 50 trials with pruning
   - Save optimization history

2. Visualize optimization results:
   - Parallel coordinate plot
   - Optimization history
   - Parameter importance
   - Best trial parameters

Run optimization for 10 trials and show me the best configuration.
```

### Task 2.3: Text-to-Sign Generation Model

**Claude Code Prompt:**
```
Implement the text-to-sign generation pipeline:

CONTEXT:
Convert text → sign gloss → motion sequence → avatar animation
This requires NLP processing and motion generation.

REQUIREMENTS:

1. Create models/sign_generation/text_processor.py:
   - Class: TextToGlossTranslator
   - Use pretrained BERT or T5
   - Fine-tune for text → gloss translation
   - Handle grammar transformation:
     * Remove articles (the, a, an)
     * Convert to topic-comment structure
     * Add temporal markers
     * Handle questions (facial expression markers)
   - Methods:
     * preprocess_text()
     * translate_to_gloss()
     * postprocess_gloss()

2. Create models/sign_generation/gloss_to_motion.py:
   - Class: GlossToMotionGenerator
   - Load sign dictionary (gloss → landmark sequence)
   - Interpolate between signs for smooth transitions
   - Methods:
     * load_sign_dictionary()
     * generate_motion_sequence()
     * smooth_transitions()
     * add_facial_expressions()

3. Create models/sign_generation/motion_smoother.py:
   - Smooth motion using:
     * Savitzky-Golay filter
     * Bezier curve interpolation
   - Ensure natural movement speed
   - Add realistic easing

4. Create frontend/src/utils/avatar_animator.js:
   - Use Three.js to render 3D avatar
   - Map landmarks to avatar skeleton
   - Implement:
     * Hand mesh deformation
     * Facial expression rigging
     * Body pose animation
   - Export animation as video or interactive 3D

5. Create scripts/generate_sign_video.py:
   - Command-line tool: text input → sign video output
   - Support multiple languages
   - Adjustable speed
   - Save as MP4

6. Create sign dictionary in data/sign_dictionary/:
   - JSON format: {gloss: {landmarks: [...], duration: float}}
   - Extract from training videos
   - Include variations for common signs

TEST:
- Test with sentences: "Hello, my name is John", "What is your name?"
- Verify smooth transitions
- Check facial expressions
- Generate sample videos

Implement this pipeline and generate 3 example videos. Show me the results.
```

**For Advanced Features:**
```
Enhance sign generation with:

1. Co-articulation modeling:
   - Signs influence each other in sequences
   - Implement transition blending based on context

2. Prosody and emphasis:
   - Allow marking words for emphasis
   - Adjust signing speed and size based on emphasis

3. Regional dialect support:
   - Load different motion patterns for dialects
   - Allow user to select dialect preference

4. Emotion integration:
   - Add emotional facial expressions
   - Adjust body language for emotion

Implement these features and show me examples with different emotions and emphasis.
```

---

## Phase 3: Backend API (Weeks 5-6)

### Task 3.1: FastAPI Application Structure

**Claude Code Prompt:**
```
Create a production-ready FastAPI backend with the following architecture:

REQUIREMENTS:

1. Create backend/api/main.py:
   - FastAPI application setup
   - CORS middleware configuration
   - API versioning (v1)
   - Health check endpoint
   - Startup/shutdown events
   - Exception handlers
   - Request ID middleware for tracing
   - Rate limiting middleware

2. Create backend/api/routes/:
   - translation.py - Sign/text translation endpoints
   - learning.py - Learning content and progress endpoints
   - user.py - User management endpoints
   - analytics.py - Usage analytics endpoints

3. Create backend/api/schemas/:
   - Pydantic models for request/response
   - translation_schemas.py:
     * TranslationRequest
     * TranslationResponse
     * SignVideoInput
     * TextInput
   - user_schemas.py:
     * UserCreate, UserUpdate, UserResponse
     * UserPreferences
   - learning_schemas.py:
     * Lesson, LessonProgress, Quiz

4. Create backend/services/:
   - translation_service.py:
     * Class: TranslationService
     * Methods:
       - async sign_to_text(video_data) → text
       - async text_to_sign(text, language) → motion_data
       - Uses model inference with caching
   - learning_engine.py:
     * Class: LearningEngine
     * Adaptive difficulty algorithm
     * Progress tracking
     * Lesson recommendation
   - user_service.py:
     * User CRUD operations
     * Authentication logic

5. Create backend/database/:
   - models.py - SQLAlchemy ORM models:
     * User
     * TranslationHistory
     * LearningProgress
     * UserPreferences
   - database.py - Database connection setup
   - crud.py - Database operations

6. Create backend/ml_serving/:
   - model_loader.py:
     * Load trained models on startup
     * Model versioning
     * Graceful fallback if model unavailable
   - inference_pipeline.py:
     * Batched inference
     * Request queuing
     * GPU memory management
   - cache.py:
     * Redis caching for common translations

7. Create backend/middleware/:
   - auth.py - JWT authentication
   - rate_limit.py - Rate limiting
   - logging.py - Request/response logging

8. Create backend/config.py:
   - Configuration management
   - Environment variables
   - Secrets management

ENDPOINTS SPECIFICATION:

POST /api/v1/translate/sign-to-text
- Input: multipart/form-data with video file
- Output: {text: str, confidence: float, processing_time: float}

POST /api/v1/translate/text-to-sign
- Input: {text: str, target_language: str, speed: float}
- Output: {motion_data: [...], video_url: str}

GET /api/v1/learning/lessons
- Query params: language, difficulty
- Output: {lessons: [...], total: int}

POST /api/v1/learning/progress
- Input: {lesson_id: str, score: float, time_spent: int}
- Output: {next_lesson: {...}, achievement: {...}}

GET /api/v1/user/preferences
- Headers: Authorization: Bearer <token>
- Output: UserPreferences

REQUIREMENTS:
- All endpoints must have OpenAPI documentation
- Input validation with Pydantic
- Proper error handling (4xx, 5xx)
- Async where possible
- Security headers
- API key authentication for public endpoints
- JWT for user-specific endpoints

Implement this backend structure and run the server. Show me:
1. API documentation at /docs
2. Test all endpoints with curl or httpx
3. Database schema visualization
```

**Testing the API:**
```
Create comprehensive API tests:

1. Create tests/test_api/:
   - test_translation.py:
     * Test sign-to-text with sample video
     * Test text-to-sign with various inputs
     * Test error cases (invalid video, too large, etc.)
   - test_learning.py:
     * Test lesson retrieval
     * Test progress tracking
     * Test adaptive difficulty
   - test_auth.py:
     * Test user registration
     * Test login/logout
     * Test JWT token validation
     * Test rate limiting

2. Use pytest with:
   - pytest-asyncio for async tests
   - httpx.AsyncClient for API calls
   - Fixtures for test data
   - Database rollback after each test

3. Create tests/load_testing/:
   - Use locust for load testing
   - Simulate concurrent users
   - Test scalability
   - Identify bottlenecks

4. Create CI/CD pipeline:
   - .github/workflows/test.yml
   - Run tests on push
   - Check code coverage (>80%)
   - Linting with ruff

Run all tests and show me:
1. Test results with coverage report
2. Load test results (requests/second, latency)
3. Any failing tests with details
```

### Task 3.2: ML Model Serving

**Claude Code Prompt:**
```
Optimize ML inference for production:

1. Create backend/ml_serving/optimized_inference.py:
   - Batch inference: accumulate requests for 100ms, process batch
   - Model quantization for faster inference
   - ONNX Runtime integration
   - TensorRT optimization (if GPU available)

2. Create backend/ml_serving/model_cache.py:
   - Cache recent translations in Redis
   - Cache common phrases
   - Implement cache invalidation
   - Track cache hit rate

3. Create backend/ml_serving/queue_manager.py:
   - Use Celery for async task processing
   - Priority queue (paid users first)
   - Retry logic for failed tasks
   - Dead letter queue

4. Benchmark performance:
   - Latency: p50, p95, p99
   - Throughput: requests/second
   - GPU utilization
   - Memory usage

5. Create monitoring dashboard:
   - Grafana dashboard config
   - Prometheus metrics
   - Alert rules

Implement and show me:
1. Performance before/after optimization
2. Cache hit rates
3. Resource utilization graphs
```

---

## Phase 4: Frontend (Weeks 7-8)

### Task 4.1: React Application Setup

**Claude Code Prompt:**
```
Create a modern React application with the following structure:

CONTEXT:
Build a responsive web app for sign language translation and learning.
Must work on desktop and mobile. Accessible (WCAG 2.2 AAA).

REQUIREMENTS:

1. Initialize React app and dependencies:
   - Create React App with TypeScript
   - Install dependencies:
     * @mui/material - UI components
     * react-router-dom - Routing
     * @tanstack/react-query - Data fetching
     * zustand - State management
     * three, @react-three/fiber - 3D rendering
     * @tensorflow/tfjs - On-device ML
     * @mediapipe/holistic - Pose detection
     * axios - API calls
     * react-webcam - Camera access

2. Create frontend/src/components/:
   - Layout/
     * Header.tsx - Navigation, user menu
     * Sidebar.tsx - Feature navigation
     * Footer.tsx - Links, credits
   - Translation/
     * VideoCapture.tsx - Webcam interface
     * TranslationView.tsx - Main translation UI
     * ResultsDisplay.tsx - Show translation results
     * LanguageSelector.tsx - Select sign language
   - Learning/
     * LessonList.tsx - Browse lessons
     * LessonPlayer.tsx - View lesson content
     * QuizInterface.tsx - Interactive quizzes
     * ProgressDashboard.tsx - User progress
   - Avatar/
     * AvatarRenderer.tsx - 3D avatar display
     * SignAnimator.tsx - Animate sign sequences
   - Common/
     * Button.tsx - Accessible button
     * Card.tsx - Content cards
     * Modal.tsx - Accessible modals
     * LoadingSpinner.tsx
     * ErrorBoundary.tsx

3. Create frontend/src/hooks/:
   - useWebcam.ts - Manage camera access
   - useMediaPipe.ts - Process video with MediaPipe
   - useTranslation.ts - API calls for translation
   - useLearning.ts - Learning content management
   - useAuth.ts - Authentication state

4. Create frontend/src/services/:
   - api.ts - Axios instance with interceptors
   - mediapipe.service.ts - MediaPipe initialization
   - translation.service.ts - Translation API calls
   - learning.service.ts - Learning API calls
   - storage.service.ts - LocalStorage/IndexedDB

5. Create frontend/src/utils/:
   - constants.ts - App constants
   - validation.ts - Input validation
   - accessibility.ts - A11y helpers
   - analytics.ts - Event tracking

6. Implement routing in App.tsx:
   - / - Home/Landing page
   - /translate - Translation interface
   - /learn - Learning dashboard
   - /learn/:lessonId - Specific lesson
   - /profile - User profile
   - /settings - App settings

7. Styling:
   - Use Material-UI theming
   - Create custom theme with high contrast
   - Responsive breakpoints
   - Dark mode support

ACCESSIBILITY REQUIREMENTS:
- All interactive elements keyboard accessible
- ARIA labels throughout
- Focus indicators
- Screen reader announcements
- Skip navigation links
- Minimum 4.5:1 contrast ratios

Implement this structure and show me:
1. Component tree visualization
2. Running app on localhost:3000
3. Basic navigation working
4. Responsive design on mobile view
```

### Task 4.2: Translation Interface

**Claude Code Prompt:**
```
Build the core translation feature with real-time video processing:

REQUIREMENTS:

1. Enhance frontend/src/components/Translation/VideoCapture.tsx:
   - Use react-webcam for camera access
   - Show live video feed (640x480)
   - Overlay MediaPipe skeleton in real-time
   - Visual feedback when sign detected
   - Support photo/video upload as alternative
   - Handle camera permissions gracefully
   - Show FPS counter

2. Create frontend/src/hooks/useSignDetection.ts:
   - Initialize MediaPipe Holistic
   - Process video frames at 30fps
   - Extract landmarks
   - Buffer frames (last 150 frames = 5 seconds)
   - Debounce inference calls (trigger after 2s of signing)
   - Cancel in-flight requests if user stops

3. Create frontend/src/components/Translation/TranslationView.tsx:
   - Split screen: Video (left) | Results (right)
   - Show detected signs in real-time
   - Display confidence scores
   - Text-to-speech for output
   - Copy to clipboard button
   - Translation history (last 10)
   - Loading states with skeleton screens

4. Implement client-side optimization:
   - Use TensorFlow.js for on-device inference (optional mode)
   - Web Workers for processing
   - Progressive enhancement (fallback to API)
   - Optimize frame rate based on device capability

5. Create frontend/src/components/Translation/TextToSignView.tsx:
   - Text input area with suggestions
   - Language selector
   - Speed control slider
   - Generate button
   - Show 3D avatar performing signs
   - Video export option
   - Share functionality

6. Add error handling:
   - Camera access denied
   - Poor lighting detection
   - Network errors
   - Model loading failures
   - Graceful degradation

PERFORMANCE TARGETS:
- First Contentful Paint: < 1.5s
- Time to Interactive: < 3s
- Smooth 30fps video processing
- < 500ms API response time (perceived)

Implement these components and show me:
1. Translation working end-to-end
2. Performance metrics from Chrome DevTools
3. Mobile responsiveness
4. Accessibility audit results
```

### Task 4.3: 3D Avatar Rendering

**Claude Code Prompt:**
```
Create a realistic 3D avatar for sign language animation:

REQUIREMENTS:

1. Create frontend/src/components/Avatar/AvatarRenderer.tsx:
   - Use @react-three/fiber and @react-three/drei
   - Load or create simple humanoid model (use Ready Player Me API or local FBX)
   - Rig with skeleton for animation
   - Materials: skin, clothing
   - Lighting setup (three-point lighting)
   - Camera controls (orbit, pan, zoom)
   - Support multiple viewing angles

2. Create frontend/src/components/Avatar/SignAnimator.tsx:
   - Map landmark coordinates to skeleton joints
   - Interpolate between keyframes
   - Smooth transitions using easing functions
   - Handle hand shapes:
     * Use shape keys/blend shapes for fingers
     * Map MediaPipe hand landmarks to finger bones
   - Facial expressions:
     * Eye blinks
     * Mouth shapes for grammar
     * Eyebrow movements
   - Implement inverse kinematics for natural movement

3. Create frontend/src/utils/animation/:
   - landmarkToSkeleton.ts - Convert landmarks to bone rotations
   - interpolation.ts - Smooth interpolation algorithms
   - ikSolver.ts - Inverse kinematics for arms
   - expressionMapper.ts - Map face landmarks to blend shapes

4. Optimize performance:
   - LOD (Level of Detail) system
   - Frustum culling
   - Texture atlasing
   - Target 60fps even on mobile

5. Add controls:
   - Play/pause animation
   - Speed control (0.5x to 2x)
   - Loop animation
   - Frame-by-frame stepping
   - Export animation as video (using MediaRecorder)

6. Create avatar customization:
   - Skin tone selector
   - Clothing options
   - Avatar size/proportions
   - Save preferences

TECHNICAL DETAILS:
- Use quaternions for rotations (avoid gimbal lock)
- Implement bone constraints
- Add subtle idle animation
- Smooth camera transitions

Implement avatar system and show me:
1. Avatar rendering with sample animation
2. FPS counter showing performance
3. Side-by-side comparison with reference video
4. Mobile performance test
```

### Task 4.4: Learning Mode

**Claude Code Prompt:**
```
Create an engaging, accessible learning experience:

REQUIREMENTS:

1. Create frontend/src/components/Learning/LessonList.tsx:
   - Display lessons grouped by difficulty
   - Show progress indicators (percentage complete)
   - Locked/unlocked states
   - Search and filter functionality
   - Recommended next lesson highlighted
   - Achievement badges

2. Create frontend/src/components/Learning/LessonPlayer.tsx:
   - Video player with controls
   - Multiple speed options (0.5x, 0.75x, 1x, 1.5x)
   - Loop specific sections
   - Step-by-step mode (frame-by-frame)
   - Side-by-side: avatar (left) | user camera (right)
   - Real-time feedback on user's signing
   - Transcript with current position highlighted

3. Create frontend/src/components/Learning/PracticeMode.tsx:
   - Show target sign
   - User performs sign
   - Real-time comparison
   - Visual feedback (green checkmark when correct)
   - Accuracy score (0-100)
   - Tips for improvement
   - Retry mechanism

4. Create frontend/src/components/Learning/QuizInterface.tsx:
   - Multiple choice questions
   - Sign recognition challenges
   - Progress bar
   - Immediate feedback
   - Explanations for incorrect answers
   - Final score and certificate

5. Implement adaptive difficulty:
   - Track user performance
   - Adjust lesson pacing
   - Skip mastered content
   - Extra practice for struggling areas
   - Spaced repetition algorithm

6. Create frontend/src/components/Learning/ProgressDashboard.tsx:
   - Total signs learned
   - Current streak
   - Practice time this week
   - Accuracy trends graph
   - Achievements earned
   - Leaderboard (optional)

7. Gamification elements:
   - Points system
   - Streak tracking
   - Achievements/badges
   - Daily challenges
   - Celebration animations

ACCESSIBILITY FOR LEARNING:
- Adjustable contrast and colors
- Text size controls
- Reduce motion option
- Keyboard shortcuts for all controls
- Audio descriptions
- Progress saving (resume anytime)

Implement learning mode and show me:
1. Complete lesson flow (browse → learn → practice → quiz)
2. Adaptive difficulty working
3. Progress persistence
4. Accessibility features demo
```

---

## Phase 5: Integration & Testing (Weeks 9-10)

### Task 5.1: End-to-End Integration

**Claude Code Prompt:**
```
Integrate all components and ensure seamless operation:

REQUIREMENTS:

1. Create integration tests:
   - tests/integration/test_full_flow.py:
     * User registration → Login → Translation → Learning → Logout
     * Test data flow through entire stack
     * Verify state consistency
   - tests/integration/test_realtime_translation.py:
     * Upload video → Process → Return translation
     * Verify latency requirements
     * Test concurrent users

2. Create frontend/src/App.test.tsx:
   - E2E tests using Playwright or Cypress
   - Test user journeys:
     * First-time user onboarding
     * Translation workflow
     * Learning a lesson
     * Settings customization

3. Create docker-compose-integration.yml:
   - Spin up entire stack (backend, frontend, database, redis)
   - Run integration tests
   - Teardown

4. Set up monitoring:
   - Sentry for error tracking (frontend + backend)
   - LogRocket for session replay
   - Google Analytics for usage metrics
   - Custom events for feature usage

5. Create health check system:
   - /health endpoint checking:
     * Database connectivity
     * Redis connectivity
     * ML model loaded
     * Disk space
     * Memory usage
   - Frontend health check (service worker)

6. Implement graceful degradation:
   - Offline mode with cached lessons
   - Progressive Web App (PWA) setup
   - Service worker for caching
   - Background sync for uploads

7. Cross-browser testing:
   - Chrome, Firefox, Safari, Edge
   - Mobile browsers (iOS Safari, Chrome Mobile)
   - Document browser support requirements

RUN COMPREHENSIVE TESTS:
Execute all integration tests and show me:
1. Test coverage report (aim for >80%)
2. Performance benchmarks
3. Browser compatibility matrix
4. Any bugs or issues found
```

### Task 5.2: User Acceptance Testing

**Claude Code Prompt:**
```
Prepare for user testing with DHH community:

REQUIREMENTS:

1. Create user testing script:
   - docs/user_testing_protocol.md:
     * Introduction and consent
     * Task scenarios
     * Success criteria
     * Survey questions
     * Accessibility evaluation

2. Create feedback collection system:
   - In-app feedback button
   - backend/api/routes/feedback.py endpoint
   - Store feedback in database
   - Email notifications for critical feedback

3. Create analytics dashboard:
   - scripts/analytics/dashboard.py:
     * User engagement metrics
     * Feature usage statistics
     * Error rate tracking
     * Performance metrics
   - Generate weekly report

4. Implement A/B testing:
   - Feature flags system
   - Test variations of UI
   - Track conversion metrics

5. Create demo video:
   - Screen recording of key features
   - Voiceover explaining functionality
   - Captions for accessibility
   - Upload to YouTube/Vimeo

6. Create user documentation:
   - docs/user_guide.md:
     * Getting started
     * Translation features
     * Learning mode
     * Troubleshooting
     * FAQ
   - Convert to HTML with mkdocs

Generate all UAT materials and show me:
1. Testing protocol PDF
2. Feedback collection system working
3. Analytics dashboard
4. User documentation website
```

---

## Phase 6: Deployment (Weeks 11-12)

### Task 6.1: Containerization

**Claude Code Prompt:**
```
Create production-ready Docker deployment:

REQUIREMENTS:

1. Create docker/Dockerfile.backend:
   - Multi-stage build
   - Base: python:3.10-slim
   - Install system dependencies
   - Copy requirements and install
   - Copy application code
   - Non-root user
   - Health check
   - Optimize layers for caching

2. Create docker/Dockerfile.frontend:
   - Multi-stage build
   - Build stage: node:18 for npm build
   - Production stage: nginx:alpine
   - Copy built files
   - Nginx config for SPA routing
   - Security headers
   - Gzip compression

3. Create docker/Dockerfile.ml-server:
   - GPU support with nvidia/cuda base
   - PyTorch with CUDA
   - Model files copied
   - Warm-up script
   - Health check endpoint

4. Create docker-compose.prod.yml:
   - Services:
     * backend (3 replicas)
     * ml-server (1 replica with GPU)
     * frontend (nginx)
     * postgres
     * redis
     * celery-worker
     * celery-beat
   - Networks configuration
   - Volume mounts
   - Environment variables
   - Resource limits
   - Restart policies

5. Create deployment scripts:
   - scripts/deployment/deploy.sh:
     * Build images
     * Tag with version
     * Push to registry
     * Deploy to server
     * Run migrations
     * Health checks
     * Rollback on failure

6. Security:
   - Non-root containers
   - Scan images for vulnerabilities (Trivy)
   - Secrets management (use Docker secrets)
   - Network isolation
   - Rate limiting

Build and test all containers locally, show me:
1. Docker build success for all images
2. docker-compose up working
3. Image sizes
4. Security scan results
```

### Task 6.2: Cloud Deployment

**Claude Code Prompt:**
```
Deploy SignBridge to cloud infrastructure (AWS/GCP):

REQUIREMENTS:

1. Infrastructure as Code (Terraform):
   - terraform/:
     * main.tf - Main configuration
     * variables.tf - Input variables
     * outputs.tf - Output values
     * backend.tf - Remote state
   - Resources:
     * VPC and subnets
     * ECS/Kubernetes cluster
     * RDS PostgreSQL instance
     * ElastiCache Redis
     * S3 buckets (videos, models, static)
     * CloudFront CDN
     * Load balancer
     * Auto-scaling groups
     * CloudWatch logs

2. Kubernetes manifests (if using K8s):
   - k8s/:
     * deployment-backend.yaml
     * deployment-frontend.yaml
     * deployment-ml.yaml
     * service.yaml
     * ingress.yaml
     * configmap.yaml
     * secrets.yaml
     * hpa.yaml (horizontal pod autoscaling)

3. CI/CD Pipeline:
   - .github/workflows/deploy.yml:
     * Trigger on push to main
     * Run tests
     * Build Docker images
     * Push to ECR/GCR
     * Deploy to staging
     * Run smoke tests
     * Deploy to production (manual approval)
     * Notify on Slack/Discord

4. Monitoring setup:
   - CloudWatch/Stackdriver dashboards
   - Alerts for:
     * High error rate
     * High latency
     * Low disk space
     * Memory leaks
     * Model inference failures
   - PagerDuty integration

5. Backup and recovery:
   - Automated database backups
   - Model versioning in S3
   - Disaster recovery plan
   - Restore testing

6. SSL/TLS:
   - Let's Encrypt certificates
   - Auto-renewal setup
   - HTTPS redirect
   - HSTS headers

7. Domain and DNS:
   - Configure Route53/Cloud DNS
   - signbridge.com → frontend
   - api.signbridge.com → backend
   - www.signbridge.com → redirect

Create infrastructure and deploy to staging, show me:
1. Terraform plan output
2. Deployed URLs
3. Health check results
4. Response times
5. Cost estimation
```

### Task 6.3: Production Optimization

**Claude Code Prompt:**
```
Optimize SignBridge for production performance and cost:

REQUIREMENTS:

1. Backend optimizations:
   - Enable response compression (gzip/brotli)
   - Implement request caching
   - Database query optimization:
     * Add indexes
     * Use select_related/prefetch_related
     * Query profiling
   - Connection pooling
   - Async everywhere possible

2. Frontend optimizations:
   - Code splitting
   - Lazy loading routes and components
   - Image optimization (WebP, responsive images)
   - Bundle size analysis
   - Tree shaking
   - Remove console.logs
   - Service worker caching strategy

3. ML inference optimizations:
   - Model quantization (FP16)
   - Batch inference
   - Model caching
   - TensorRT optimization
   - Queue management

4. CDN configuration:
   - Cache static assets (1 year)
   - Cache API responses (5 minutes)
   - Edge caching for videos
   - Invalidation strategy

5. Cost optimization:
   - Right-size instances
   - Use spot/preemptible instances for ML
   - Auto-scaling policies
   - Reserved instances for baseline
   - S3 lifecycle policies
   - CloudFront request pricing

6. Performance monitoring:
   - Real User Monitoring (RUM)
   - Synthetic monitoring
   - Core Web Vitals tracking
   - Lighthouse CI integration

Implement optimizations and show me:
1. Before/after performance comparison
2. Lighthouse scores (aim for >90 all categories)
3. Bundle size reduction
4. Cost savings projection
5. Load test results (1000 concurrent users)
```

---

## Prompt Engineering Tips

### How to Write Effective Prompts for Claude Code

#### 1. **Be Specific and Detailed**

❌ Bad:
```
Create a video processing script
```

✅ Good:
```
Create scripts/data_preprocessing/video_processor.py with:
- Class: VideoProcessor
- Extract frames at 30fps using OpenCV
- Resize to 640x480
- Save frames as JPEG to data/frames/
- Generate metadata JSON with frame count, duration, fps
- Add progress bar with tqdm
- Handle errors: file not found, corrupted video, out of memory
- Log all actions to logs/video_processing.log
```

#### 2. **Provide Context**

Always explain:
- **What** you're building
- **Why** this component is needed
- **How** it fits into the larger system

Example:
```
CONTEXT:
We're building a sign language recognition system. This component processes 
raw video files to extract pose landmarks using MediaPipe. It's the first 
step in our data pipeline and must handle 10,000+ videos efficiently.

REQUIREMENTS:
[your requirements here]
```

#### 3. **Specify File Locations**

❌ Bad:
```
Create a model file
```

✅ Good:
```
Create models/sign_recognition/spatial_encoder.py
```

#### 4. **Request Tests and Execution**

Always end with:
```
Implement this and:
1. Run the code
2. Show me the output
3. Run tests if applicable
4. Report any errors
```

#### 5. **Use Iterative Refinement**

If Claude's implementation isn't perfect:
```
Good start! Now enhance it by:
1. Adding error handling for [specific case]
2. Improving performance by [specific technique]
3. Adding logging at [specific points]

Run the updated version and show me the improvements.
```

#### 6. **Ask for Explanations**

```
Explain the trade-offs between using:
1. Transformer vs LSTM for temporal modeling
2. Which would be better for our use case with sequences up to 150 frames?

Then implement the better option.
```

### Debugging with Claude Code

When things go wrong:

**Strategy 1: Detailed Error Investigation**
```
The model training is failing with CUDA out of memory. Debug by:
1. Print model size and memory usage
2. Print batch size and sequence lengths
3. Try reducing batch size from 32 to 16
4. Implement gradient accumulation
5. Add memory profiling

Show me memory usage at each step.
```

**Strategy 2: Incremental Testing**
```
The pipeline is breaking somewhere. Test each component:
1. Test landmark extraction on single video - does it work?
2. Test feature engineering on extracted landmarks - does it work?
3. Test dataset loading - does it work?
4. Test model forward pass with dummy data - does it work?

Show me results for each test and identify where it breaks.
```

**Strategy 3: Comparison with Known Good**
```
Create a minimal reproducible example:
1. Use a tiny dataset (5 videos)
2. Use a simple model (1 layer)
3. Train for 2 epochs
4. If this works, gradually add complexity
5. Find what causes the failure

Show me results at each step.
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: MediaPipe Installation Fails

**Claude Code Prompt:**
```
MediaPipe installation is failing. Try these solutions in order:
1. Install system dependencies for MediaPipe
2. Try pip install mediapipe==0.10.9 (specific version)
3. Build from source if necessary
4. Create a Docker container with working MediaPipe

Document which solution works and update requirements.txt accordingly.
```

#### Issue 2: Model Training Diverges

**Claude Code Prompt:**
```
Training loss is exploding (NaN values). Debug and fix:
1. Add gradient clipping (max_grad_norm=1.0)
2. Reduce learning rate by 10x
3. Add batch normalization if missing
4. Check for data issues (NaN in inputs)
5. Implement learning rate warmup
6. Add gradient monitoring to detect explosion early

Implement fixes and re-run training. Show me loss curves.
```

#### Issue 3: Frontend Performance Issues

**Claude Code Prompt:**
```
Frontend is laggy when processing video. Profile and optimize:
1. Use Chrome DevTools Performance tab
2. Identify bottlenecks
3. Move heavy processing to Web Worker
4. Reduce MediaPipe processing frequency (every 2nd frame)
5. Debounce API calls
6. Implement request cancellation

Show me before/after FPS measurements.
```

#### Issue 4: CORS Errors

**Claude Code Prompt:**
```
Getting CORS errors in browser. Fix by:
1. Add CORS middleware to FastAPI
2. Configure allowed origins
3. Add preflight request handling
4. Set credentials: true if needed

Test from frontend and show me network tab showing successful requests.
```

#### Issue 5: Database Connection Errors

**Claude Code Prompt:**
```
Can't connect to PostgreSQL. Troubleshoot:
1. Verify PostgreSQL is running (ps aux | grep postgres)
2. Check connection string format
3. Verify credentials
4. Check firewall rules
5. Test connection with psql CLI
6. Check Docker network if using containers

Show me successful connection test.
```

---

## Advanced Claude Code Workflows

### Workflow 1: Feature Development

```bash
# Start new feature branch
git checkout -b feature/learning-mode

# Use Claude Code to implement
claude-code run "
Implement the learning mode feature as specified in:
- docs/learning_mode_spec.md

Create:
1. Backend endpoints
2. Frontend components  
3. Tests
4. Documentation

Run all tests and show me results.
"

# Review changes
git diff

# Commit
git add .
git commit -m "feat: implement learning mode"
```

### Workflow 2: Bug Fixing

```bash
# Claude Code can help debug
claude-code run "
Users report that sign detection is failing for left-handed signers.
Debug by:
1. Reviewing landmark extraction code
2. Checking if left/right hand detection is flipped
3. Testing with left-handed sample videos
4. Fixing the issue
5. Adding test case for left-handed signers

Show me the fix and test results.
"
```

### Workflow 3: Refactoring

```bash
claude-code run "
Refactor models/sign_recognition/ to improve maintainability:
1. Extract common code into utils
2. Add type hints throughout
3. Improve docstrings
4. Split large functions
5. Add more unit tests
6. Ensure all tests still pass

Show me diff and test results.
"
```

### Workflow 4: Documentation

```bash
claude-code run "
Generate comprehensive documentation:
1. API documentation in docs/api/
2. Architecture diagrams (use mermaid)
3. Setup guide
4. Deployment guide
5. Troubleshooting guide
6. Contributing guidelines

Use mkdocs for rendering. Show me the generated docs site.
"
```

---

## Final Integration Prompt

When you've completed all phases and want to test everything together:

```
Run end-to-end integration test of SignBridge:

1. Start all services (Docker Compose)
2. Verify all health checks pass
3. Test sign-to-text translation:
   - Use sample ASL video
   - Verify correct translation
   - Check response time < 2 seconds
4. Test text-to-sign generation:
   - Input: "Hello, my name is Sarah"
   - Verify avatar animation is smooth
5. Test learning mode:
   - Load lesson
   - Complete practice exercise
   - Verify progress saved
6. Run load test:
   - 100 concurrent users
   - Verify no errors
   - Check p95 latency
7. Test on mobile:
   - iOS Safari
   - Android Chrome
8. Accessibility audit:
   - Run axe DevTools
   - Verify WCAG 2.2 AA compliance

Generate comprehensive test report with:
- All test results
- Performance metrics
- Screenshots
- Known issues
- Recommendations

Show me the report and summary of any failures.
```

---

## Cost Estimation

### Using Claude Code Effectively

**Estimated Claude Code Usage:**

| Phase | Estimated Prompts | Token Usage | Approximate Cost* |
|-------|------------------|-------------|-------------------|
| Setup | 10-15 | 500K | $5 |
| Data Pipeline | 30-40 | 1.5M | $15 |
| ML Models | 50-60 | 2M | $20 |
| Backend | 40-50 | 1.5M | $15 |
| Frontend | 50-60 | 2M | $20 |
| Integration | 30-40 | 1M | $10 |
| Deployment | 20-30 | 800K | $8 |
| **Total** | **230-295** | **9.3M** | **~$93** |

*Approximate costs based on Sonnet 4.5 pricing. Actual costs may vary.

### Tips to Reduce Costs:

1. **Be Specific**: Detailed prompts reduce back-and-forth
2. **Batch Requests**: Combine related tasks in one prompt
3. **Use Context Files**: Reference existing code instead of repeating it
4. **Iterate Smartly**: Fix specific issues rather than regenerating entire files
5. **Cache Results**: Save successful implementations locally

---

## Summary

This guide provides everything needed to build SignBridge using Claude Code:

✅ **Complete project structure** with detailed file-by-file instructions
✅ **Phase-by-phase implementation** plan over 12 weeks
✅ **Specific Claude Code prompts** for each component
✅ **Testing and debugging** strategies
✅ **Deployment and optimization** guidance
✅ **Troubleshooting** common issues

### Next Steps:

1. **Week 1**: Start with project setup and data collection
2. **Week 2**: Get first model training
3. **Week 3**: Build basic API
4. **Week 4**: Create MVP frontend
5. **Week 5+**: Iterate and enhance

Remember: Claude Code works best with **clear, detailed prompts** and **iterative refinement**. Don't try to build everything at once—break it into manageable pieces and test frequently.

Good luck building SignBridge! This project has the potential to make a real impact on accessibility and communication. 🚀
