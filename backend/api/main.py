"""
SignBridge FastAPI Application
Enhanced backend for ASL sign recognition with MediaPipe, learning mode, and sentence translation
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from PIL import Image
import io
import sys
from pathlib import Path
import cv2
import mediapipe as mp
from datetime import datetime, timedelta
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.simple_cnn import SimpleCNN
from backend.ml_serving.model_loader import ModelLoader

# Create FastAPI app
app = FastAPI(
    title="SignBridge API",
    description="ASL Sign Recognition API",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model loader
model_loader = ModelLoader()
model = None  # Keep for backward compatibility
device = None
class_labels = None

# ASL MNIST class labels (A-Z excluding J and Z)
ASL_LABELS = [chr(i) for i in range(ord('A'), ord('Z')+1) if chr(i) not in ['J', 'Z']]

# Initialize MediaPipe Hands (optional, only used if use_mediapipe=True)
# Note: MediaPipe is disabled by default as the model was trained on full-frame images
# To enable MediaPipe, you'll need to download the hand landmark model from:
# https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
hands = None
try:
    # Use new MediaPipe API (0.10+)
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from pathlib import Path
    
    # Try to find model file, but don't fail if not found
    model_path = project_root / 'models' / 'mediapipe' / 'hand_landmarker.task'
    if model_path.exists():
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        hands = vision.HandLandmarker.create_from_options(options)
        print("MediaPipe HandLandmarker initialized successfully")
    else:
        print("MediaPipe model not found. Hand detection disabled. (This is OK - using standard preprocessing)")
except (AttributeError, ImportError, Exception) as e:
    print(f"MediaPipe not available: {e}. Using standard preprocessing (recommended for current model).")
    hands = None

# Sentence buffer for multi-sign translation
prediction_buffer = defaultdict(lambda: {'signs': [], 'last_update': datetime.now()})

# Learning mode content
LEARNING_CONTENT = {
    'A': {
        'description': 'Closed fist with thumb on the side',
        'tips': ['Keep fingers closed tightly', 'Thumb should touch the side of your fist', 'Palm faces forward'],
        'common_mistakes': ['Thumb sticking up or out', 'Fingers not fully closed', 'Fist too loose']
    },
    'B': {
        'description': 'Flat hand, fingers together, thumb across palm',
        'tips': ['Keep all 4 fingers straight and together', 'Tuck thumb across palm', 'Palm faces forward'],
        'common_mistakes': ['Fingers spread apart', 'Thumb not tucked in', 'Bent fingers']
    },
    'C': {
        'description': 'Curved hand forming the letter C',
        'tips': ['Curve all fingers uniformly', 'Keep thumb curved too', 'Leave space to show the C shape'],
        'common_mistakes': ['Fingers too straight', 'Not curved enough', 'Thumb position wrong']
    },
    'D': {
        'description': 'Index finger up, other fingers touching thumb',
        'tips': ['Point index finger straight up', 'Middle, ring, pinky touch thumb', 'Form a circle with thumb'],
        'common_mistakes': ['Index finger bent', 'Other fingers not touching thumb', 'Thumb not forming circle']
    },
    'E': {
        'description': 'Fingers curled down, thumb across',
        'tips': ['Curl all fingers inward', 'Thumb across the fingers', 'Tight fist position'],
        'common_mistakes': ['Fingers not curled enough', 'Thumb position incorrect', 'Too loose']
    },
    'F': {
        'description': 'Index and thumb forming circle, other fingers up',
        'tips': ['Touch index finger tip to thumb tip', 'Keep other 3 fingers straight up', 'Circle should be clear'],
        'common_mistakes': ['Circle not closed', 'Other fingers bent', 'Index finger wrong position']
    },
    'G': {
        'description': 'Index finger and thumb pointing sideways',
        'tips': ['Point index finger sideways', 'Thumb parallel to index', 'Fist closed for other fingers'],
        'common_mistakes': ['Fingers not parallel', 'Thumb position wrong', 'Other fingers not closed']
    },
    'H': {
        'description': 'Index and middle fingers together, pointing sideways',
        'tips': ['Extend index and middle together', 'Point sideways', 'Keep fingers touching'],
        'common_mistakes': ['Fingers spread apart', 'Not pointing sideways', 'Other fingers not closed']
    },
    'I': {
        'description': 'Pinky finger up, fist closed',
        'tips': ['Extend only pinky finger', 'Keep it straight', 'Close all other fingers'],
        'common_mistakes': ['Pinky bent', 'Other fingers not fully closed', 'Thumb sticking out']
    },
    'K': {
        'description': 'Index and middle up in V, thumb touches middle finger',
        'tips': ['Form V with index and middle', 'Thumb touches middle finger', 'Other fingers closed'],
        'common_mistakes': ['Thumb not touching middle finger', 'V shape unclear', 'Fingers bent']
    },
    'L': {
        'description': 'L shape with index and thumb',
        'tips': ['Index finger straight up', 'Thumb straight out', 'Form 90-degree angle'],
        'common_mistakes': ['Angle not 90 degrees', 'Fingers bent', 'Other fingers not closed']
    },
    'M': {
        'description': 'Three fingers over thumb',
        'tips': ['Drape index, middle, ring over thumb', 'Pinky tucked', 'Clear M shape'],
        'common_mistakes': ['Wrong number of fingers', 'Thumb position unclear', 'Fingers not draped']
    },
    'N': {
        'description': 'Two fingers over thumb',
        'tips': ['Drape index and middle over thumb', 'Other fingers closed', 'Similar to M but 2 fingers'],
        'common_mistakes': ['Three fingers instead of two', 'Thumb not visible', 'Fingers bent']
    },
    'O': {
        'description': 'All fingertips touching thumb forming O',
        'tips': ['Touch all fingertips to thumb', 'Form round circle', 'Keep circle clear'],
        'common_mistakes': ['Circle not closed', 'Fingers not all touching', 'Not round enough']
    },
    'P': {
        'description': 'K sign but pointing downward',
        'tips': ['Form K shape', 'Point downward at angle', 'Thumb touches middle finger'],
        'common_mistakes': ['Angle wrong', 'Looks like K', 'Thumb position incorrect']
    },
    'Q': {
        'description': 'G sign but pointing downward',
        'tips': ['Form G shape', 'Point downward', 'Thumb and index finger clear'],
        'common_mistakes': ['Pointing sideways instead of down', 'Looks like G', 'Fingers not clear']
    },
    'R': {
        'description': 'Index and middle crossed',
        'tips': ['Cross index over middle', 'Keep fingers straight', 'Other fingers closed'],
        'common_mistakes': ['Fingers not crossed', 'Fingers bent', 'Other fingers open']
    },
    'S': {
        'description': 'Fist with thumb across fingers',
        'tips': ['Make tight fist', 'Thumb across front', 'Cover all fingers'],
        'common_mistakes': ['Thumb on side instead of front', 'Fist too loose', 'Thumb not covering']
    },
    'T': {
        'description': 'Thumb between index and middle',
        'tips': ['Insert thumb between index and middle', 'Make fist', 'Thumb should poke through'],
        'common_mistakes': ['Thumb not visible', 'Wrong finger position', 'Fist not closed']
    },
    'U': {
        'description': 'Index and middle together, pointing up',
        'tips': ['Extend index and middle up', 'Keep them touching', 'Straight up, not sideways'],
        'common_mistakes': ['Fingers apart', 'Pointing sideways (that\'s H)', 'Fingers bent']
    },
    'V': {
        'description': 'Index and middle apart in V shape',
        'tips': ['Spread index and middle in V', 'Keep fingers straight', 'Clear V shape'],
        'common_mistakes': ['Fingers together (that\'s U)', 'Fingers bent', 'V angle too wide or narrow']
    },
    'W': {
        'description': 'Three fingers up in W shape',
        'tips': ['Extend index, middle, ring fingers', 'Spread them apart', 'Keep straight'],
        'common_mistakes': ['Fingers together', 'Pinky up too (that\'s 4)', 'Fingers bent']
    },
    'X': {
        'description': 'Index finger crooked like a hook',
        'tips': ['Bend index finger at first knuckle', 'Form hook shape', 'Other fingers closed'],
        'common_mistakes': ['Finger too straight', 'Bent at wrong joint', 'Other fingers open']
    },
    'Y': {
        'description': 'Thumb and pinky extended, shaka sign',
        'tips': ['Extend thumb and pinky', 'Keep them spread', 'Close other fingers'],
        'common_mistakes': ['Other fingers not closed', 'Thumb or pinky bent', 'Not spread enough']
    }
}


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: str
    confidence: float
    top_3: list
    success: bool


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, device, class_labels, model_loader

    print("Loading model...")
    
    # Try to load using new model loader (supports both simple CNN and spatial-temporal)
    success = model_loader.load_model(model_type='auto')
    
    if success:
        model = model_loader.model  # For backward compatibility
        device = model_loader.device
        class_labels = model_loader.class_labels
        
        print(f"Model type: {model_loader.model_type}")
        print(f"Classes: {class_labels[:10]}...")
    else:
        # Fallback to old loading method
        print("Trying fallback model loading...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        checkpoint_path = project_root / 'models' / 'checkpoints' / 'best_model.pth'

        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            num_classes = checkpoint.get('num_classes', 24)

            model = SimpleCNN(num_classes=num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()

            val_acc = checkpoint.get('val_acc', 0)
            print(f"Model loaded successfully! Validation accuracy: {val_acc:.2f}%")

            class_labels = ASL_LABELS[:num_classes]
            print(f"Classes: {class_labels}")
        else:
            print(f"Warning: Model checkpoint not found")
            print("Please train the model first using: python scripts/training/train_simple_model.py")


def preprocess_image(image: Image.Image, use_mediapipe: bool = True) -> torch.Tensor:
    """
    Preprocess image for model input with optional MediaPipe hand detection

    Args:
        image: PIL Image
        use_mediapipe: Whether to use MediaPipe for hand detection and cropping

    Returns:
        Preprocessed tensor
    """
    if use_mediapipe and hands is not None:
        try:
            # Convert PIL to numpy array (RGB)
            img_array = np.array(image.convert('RGB'))
            
            # Create MediaPipe Image
            from mediapipe import Image as MPImage
            mp_image = MPImage(image_format=MPImage.ImageFormat.SRGB, data=img_array)
            
            # Detect hand using new API
            detection_result = hands.detect(mp_image)
            
            if detection_result.hand_landmarks:
                # Get hand bounding box from first hand
                h, w = img_array.shape[:2]
                landmarks = detection_result.hand_landmarks[0]
                
                x_coords = [lm.x * w for lm in landmarks]
                y_coords = [lm.y * h for lm in landmarks]
                
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                # Add padding (20%)
                padding_x = int((x_max - x_min) * 0.2)
                padding_y = int((y_max - y_min) * 0.2)
                
                x_min = max(0, x_min - padding_x)
                y_min = max(0, y_min - padding_y)
                x_max = min(w, x_max + padding_x)
                y_max = min(h, y_max + padding_y)
                
                # Crop to hand region
                hand_crop = img_array[y_min:y_max, x_min:x_max]
                
                if hand_crop.size > 0:
                    # Convert to grayscale
                    if len(hand_crop.shape) == 3:
                        hand_gray = cv2.cvtColor(hand_crop, cv2.COLOR_RGB2GRAY)
                    else:
                        hand_gray = hand_crop
                    
                    # Resize to 28x28
                    hand_resized = cv2.resize(hand_gray, (28, 28))
                    
                    # Normalize to [0, 1]
                    img_array = hand_resized.astype(np.float32) / 255.0
                    
                    # Add batch and channel dimensions
                    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
                    
                    return img_tensor
        except Exception as e:
            print(f"MediaPipe processing error: {e}, falling back to standard preprocessing")

    # Fallback to original preprocessing if MediaPipe fails or is disabled
    # Match training data preprocessing: convert to grayscale, resize to 28x28, normalize
    image = image.convert('L')
    
    # Resize using LANCZOS for better quality (matches PIL default)
    image = image.resize((28, 28), Image.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    
    # Normalize to [0, 1] (matching training data)
    img_array = img_array / 255.0
    
    # Add batch and channel dimensions: (28, 28) -> (1, 1, 28, 28)
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)

    return img_tensor


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SignBridge API",
        "version": "0.1.0",
        "status": "running",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "not set",
        "classes": len(class_labels) if class_labels else 0
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict ASL sign from uploaded image

    Args:
        file: Uploaded image file

    Returns:
        Prediction response with top prediction and confidence
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )

    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Preprocess image (disable MediaPipe to match training data format)
        img_tensor = preprocess_image(image, use_mediapipe=False)
        img_tensor = img_tensor.to(device)

        # Run inference using model loader if available, otherwise fallback
        if model_loader.model is not None:
            # Use new model loader
            result = model_loader.predict(img_tensor)
            return PredictionResponse(
                prediction=result['prediction'],
                confidence=result['confidence'],
                top_3=result['top_3'],
                success=True
            )
        else:
            # Fallback to old method
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)

                confidence, predicted_idx = probabilities.max(1)
                predicted_class = class_labels[predicted_idx.item()]
                confidence_score = confidence.item()

                top3_probs, top3_indices = probabilities.topk(3, dim=1)
                top3_predictions = [
                    {
                        "class": class_labels[idx.item()],
                        "confidence": prob.item()
                    }
                    for prob, idx in zip(top3_probs[0], top3_indices[0])
                ]

            return PredictionResponse(
                prediction=predicted_class,
                confidence=confidence_score,
                top_3=top3_predictions,
                success=True
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.get("/classes")
async def get_classes():
    """Get list of supported classes"""
    if class_labels is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "classes": class_labels,
        "count": len(class_labels)
    }


@app.get("/learn/{letter}")
async def get_learning_content(letter: str):
    """Get learning content for a specific letter"""
    letter = letter.upper()
    if letter in LEARNING_CONTENT:
        return LEARNING_CONTENT[letter]
    raise HTTPException(status_code=404, detail="Letter not found")


@app.post("/practice/{letter}")
async def practice_letter(letter: str, file: UploadFile = File(...)):
    """Practice mode - returns detailed feedback for learning"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Get prediction
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        img_tensor = preprocess_image(image)
        img_tensor = img_tensor.to(device)

        # Use model loader if available
        if model_loader.model is not None:
            result = model_loader.predict(img_tensor)
            predicted_class = result['prediction']
            confidence_score = result['confidence']
            top3_predictions = result['top_3']
        else:
            # Fallback
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = probabilities.max(1)
                predicted_class = class_labels[predicted_idx.item()]
                confidence_score = confidence.item()

                top3_probs, top3_indices = probabilities.topk(3, dim=1)
                top3_predictions = [
                    {"class": class_labels[idx.item()], "confidence": prob.item()}
                    for prob, idx in zip(top3_probs[0], top3_indices[0])
                ]

        # Compare with target letter
        target_letter = letter.upper()
        is_correct = predicted_class == target_letter

        feedback = {
            'correct': is_correct,
            'target': target_letter,
            'your_sign': predicted_class,
            'confidence': confidence_score,
            'top_3': top3_predictions,
            'feedback_message': ''
        }

        if is_correct:
            if confidence_score > 0.9:
                feedback['feedback_message'] = 'Perfect! Excellent form!'
            elif confidence_score > 0.7:
                feedback['feedback_message'] = 'Good job! Try to be more precise.'
            else:
                feedback['feedback_message'] = 'Correct, but could be clearer.'
        else:
            feedback['feedback_message'] = f'That looks like {predicted_class}, not {target_letter}. Check the tips and try again!'

        return feedback

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/predict_sentence")
async def predict_sentence(file: UploadFile = File(...), session_id: str = "default", buffer_time: int = 10):
    """
    Predict and buffer signs to form sentences
    Clear buffer after buffer_time seconds of inactivity
    """
    global prediction_buffer

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Get prediction
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        img_tensor = preprocess_image(image)
        img_tensor = img_tensor.to(device)

        # Use model loader if available
        if model_loader.model is not None:
            result = model_loader.predict(img_tensor)
            predicted_class = result['prediction']
            confidence_score = result['confidence']
            top3_predictions = result['top_3']
        else:
            # Fallback
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = probabilities.max(1)
                predicted_class = class_labels[predicted_idx.item()]
                confidence_score = confidence.item()

                top3_probs, top3_indices = probabilities.topk(3, dim=1)
                top3_predictions = [
                    {"class": class_labels[idx.item()], "confidence": prob.item()}
                    for prob, idx in zip(top3_probs[0], top3_indices[0])
                ]

        # Initialize buffer for this session
        if session_id not in prediction_buffer:
            prediction_buffer[session_id] = {
                'signs': [],
                'last_update': datetime.now()
            }

        buffer = prediction_buffer[session_id]

        # Clear if inactive for too long
        if datetime.now() - buffer['last_update'] > timedelta(seconds=buffer_time):
            buffer['signs'] = []

        # Add new sign if different from last (avoid duplicates)
        if not buffer['signs'] or buffer['signs'][-1] != predicted_class:
            buffer['signs'].append(predicted_class)

        buffer['last_update'] = datetime.now()

        # Form sentence
        sentence = ''.join(buffer['signs'])

        return {
            'letter': predicted_class,
            'confidence': confidence_score,
            'sentence': sentence,
            'word': sentence,
            'sign_count': len(buffer['signs']),
            'top_3': top3_predictions,
            'success': True
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/clear_sentence/{session_id}")
async def clear_sentence(session_id: str):
    """Clear the sentence buffer for a session"""
    if session_id in prediction_buffer:
        prediction_buffer[session_id] = {
            'signs': [],
            'last_update': datetime.now()
        }
    return {"status": "cleared", "session_id": session_id}


@app.get("/landmarks")
async def get_hand_landmarks(file: UploadFile = File(...)):
    """Get MediaPipe hand landmarks for visualization"""
    if hands is None:
        raise HTTPException(status_code=503, detail="MediaPipe not available")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert PIL to numpy array (RGB)
        img_array = np.array(image.convert('RGB'))
        
        # Create MediaPipe Image
        from mediapipe import Image as MPImage
        mp_image = MPImage(image_format=MPImage.ImageFormat.SRGB, data=img_array)
        
        # Detect hand using new API
        detection_result = hands.detect(mp_image)

        if detection_result.hand_landmarks:
            landmarks = detection_result.hand_landmarks[0]
            landmark_list = [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for lm in landmarks
            ]
            return {"landmarks": landmark_list, "detected": True}

        return {"landmarks": [], "detected": False}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
