"""
MediaPipe-based feature extraction for sign language recognition
Supports both Holistic (full body) and Hands-only modes
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, Dict
from enum import Enum


class FeatureMode(Enum):
    """Feature extraction modes"""
    HOLISTIC = "holistic"  # Full body + hands + face (1662 dims)
    HANDS = "hands"        # Hands only (126 dims for 2 hands, 63 for 1 hand)


class FeatureExtractor:
    """
    MediaPipe-based feature extractor for sign language recognition
    
    Based on best practices from:
    - Realtime-Sign-Language-Detection (Holistic)
    - Sign-Language-Recognition-System (Hands)
    """
    
    def __init__(
        self,
        mode: FeatureMode = FeatureMode.HOLISTIC,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize feature extractor
        
        Args:
            mode: Feature extraction mode (HOLISTIC or HANDS)
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
        """
        self.mode = mode
        
        if mode == FeatureMode.HOLISTIC:
            self.mp_holistic = mp.solutions.holistic
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=2,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.feature_dim = 1662  # 33*4 + 468*3 + 21*3 + 21*3
        else:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                max_num_hands=2
            )
            self.feature_dim = 126  # 21*3 * 2 hands
    
    def extract_landmarks(self, image: np.ndarray) -> np.ndarray:
        """
        Extract landmarks from a single image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Normalized feature vector
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        if self.mode == FeatureMode.HOLISTIC:
            results = self.holistic.process(image_rgb)
            return self._extract_holistic_features(results)
        else:
            results = self.hands.process(image_rgb)
            return self._extract_hands_features(results)
    
    def _extract_holistic_features(self, results) -> np.ndarray:
        """
        Extract features from MediaPipe Holistic results
        
        Returns:
            Feature vector of shape (1662,)
        """
        features = []
        
        # Pose landmarks (33 points × 4 = 132)
        if results.pose_landmarks:
            pose = np.array([
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in results.pose_landmarks.landmark
            ]).flatten()
            features.extend(pose)
        else:
            features.extend([0.0] * 132)
        
        # Face landmarks (468 points × 3 = 1404)
        if results.face_landmarks:
            face = np.array([
                [lm.x, lm.y, lm.z]
                for lm in results.face_landmarks.landmark
            ]).flatten()
            features.extend(face)
        else:
            features.extend([0.0] * 1404)
        
        # Left hand (21 points × 3 = 63)
        if results.left_hand_landmarks:
            lh = np.array([
                [lm.x, lm.y, lm.z]
                for lm in results.left_hand_landmarks.landmark
            ]).flatten()
            features.extend(lh)
        else:
            features.extend([0.0] * 63)
        
        # Right hand (21 points × 3 = 63)
        if results.right_hand_landmarks:
            rh = np.array([
                [lm.x, lm.y, lm.z]
                for lm in results.right_hand_landmarks.landmark
            ]).flatten()
            features.extend(rh)
        else:
            features.extend([0.0] * 63)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_hands_features(self, results) -> np.ndarray:
        """
        Extract features from MediaPipe Hands results
        
        Returns:
            Feature vector of shape (126,) for 2 hands or (63,) for 1 hand
        """
        features = []
        
        # Left hand (21 points × 3 = 63)
        if results.multi_hand_landmarks:
            # Find left and right hands
            left_hand = None
            right_hand = None
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_type = results.multi_handedness[idx].classification[0].label
                landmarks = np.array([
                    [lm.x, lm.y, lm.z]
                    for lm in hand_landmarks.landmark
                ]).flatten()
                
                if hand_type == "Left":
                    left_hand = landmarks
                else:
                    right_hand = landmarks
            
            # Add left hand
            if left_hand is not None:
                features.extend(left_hand)
            else:
                features.extend([0.0] * 63)
            
            # Add right hand
            if right_hand is not None:
                features.extend(right_hand)
            else:
                features.extend([0.0] * 63)
        else:
            # No hands detected
            features.extend([0.0] * 126)
        
        return np.array(features, dtype=np.float32)
    
    def extract_sequence(self, video_path: str) -> np.ndarray:
        """
        Extract landmarks from video sequence
        
        Args:
            video_path: Path to video file
            
        Returns:
            Array of shape (num_frames, feature_dim)
        """
        cap = cv2.VideoCapture(video_path)
        landmarks_sequence = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks = self.extract_landmarks(frame)
            landmarks_sequence.append(landmarks)
        
        cap.release()
        return np.array(landmarks_sequence)
    
    def draw_landmarks(self, image: np.ndarray, results) -> np.ndarray:
        """
        Draw landmarks on image (for visualization)
        
        Args:
            image: Input image
            results: MediaPipe results
            
        Returns:
            Image with landmarks drawn
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.mode == FeatureMode.HOLISTIC:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(
                image_rgb, results.face_landmarks,
                mp.solutions.holistic.FACEMESH_TESSELATION
            )
            mp_drawing.draw_landmarks(
                image_rgb, results.pose_landmarks,
                mp.solutions.holistic.POSE_CONNECTIONS
            )
            mp_drawing.draw_landmarks(
                image_rgb, results.left_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS
            )
            mp_drawing.draw_landmarks(
                image_rgb, results.right_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS
            )
        else:
            mp_drawing = mp.solutions.drawing_utils
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image_rgb, hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS
                    )
        
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'holistic'):
            self.holistic.close()
        if hasattr(self, 'hands'):
            self.hands.close()


