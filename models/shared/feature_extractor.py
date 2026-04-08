"""
Feature engineering utilities for sign language recognition
"""
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter
from typing import Optional


class SignFeatureEngineer:
    """Extract higher-level features from raw landmarks"""
    
    @staticmethod
    def compute_hand_shape_features(hand_landmarks: np.ndarray) -> np.ndarray:
        """
        Compute hand shape descriptors
        - Finger extensions
        - Palm orientation
        - Hand openness
        
        Args:
            hand_landmarks: Hand landmarks (21, 3) or flattened (63,)
            
        Returns:
            Feature vector (10 dimensions)
        """
        if hand_landmarks is None or len(hand_landmarks) == 0:
            return np.zeros(10, dtype=np.float32)
        
        # Reshape to (21, 3) if flattened
        if hand_landmarks.shape == (63,):
            hand_landmarks = hand_landmarks.reshape(21, 3)
        elif len(hand_landmarks.shape) != 2:
            return np.zeros(10, dtype=np.float32)
        
        features = []
        
        # Finger tip distances from palm center (wrist)
        wrist = hand_landmarks[0]
        fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        
        for tip_idx in fingertips:
            if tip_idx < len(hand_landmarks):
                dist = euclidean(hand_landmarks[tip_idx], wrist)
                features.append(dist)
            else:
                features.append(0.0)
        
        # Hand openness (average distance between adjacent fingers)
        openness = 0.0
        valid_pairs = 0
        for i in range(len(fingertips) - 1):
            if fingertips[i] < len(hand_landmarks) and fingertips[i+1] < len(hand_landmarks):
                openness += euclidean(hand_landmarks[fingertips[i]], 
                                    hand_landmarks[fingertips[i+1]])
                valid_pairs += 1
        features.append(openness / max(valid_pairs, 1))
        
        # Palm orientation (using normal vector)
        # Define palm plane with wrist, index MCP, pinky MCP
        if len(hand_landmarks) >= 18:
            p1 = hand_landmarks[0]   # Wrist
            p2 = hand_landmarks[5]   # Index MCP
            p3 = hand_landmarks[17]  # Pinky MCP
            
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm > 1e-8:
                normal = normal / norm
            else:
                normal = np.array([0.0, 0.0, 1.0])
            
            features.extend(normal)  # 3 features
        else:
            features.extend([0.0, 0.0, 1.0])
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def compute_motion_features(landmarks_sequence: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Compute temporal motion features
        - Velocity (first derivative)
        - Acceleration (second derivative)
        
        Args:
            landmarks_sequence: Sequence of landmarks (num_frames, feature_dim)
            window: Window size for smoothing
            
        Returns:
            Enhanced features with motion (num_frames, feature_dim * 3)
        """
        if len(landmarks_sequence) < window:
            # Pad with zeros if too short
            padded = np.pad(landmarks_sequence, ((0, window - len(landmarks_sequence)), (0, 0)), 'edge')
            landmarks_sequence = padded
        
        # Smooth with Savitzky-Golay filter
        try:
            smoothed = savgol_filter(landmarks_sequence, window, 3, axis=0)
        except:
            smoothed = landmarks_sequence
        
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
    def compute_spatial_features(landmarks_frame: np.ndarray) -> np.ndarray:
        """
        Compute spatial relationships between body parts
        - Hand-to-face distance
        - Hand-to-hand distance
        - Hand position relative to body center
        
        Args:
            landmarks_frame: Single frame landmarks (408 dimensions)
            
        Returns:
            Spatial feature vector (5 dimensions)
        """
        # Extract components (assuming 408-dim vector)
        if len(landmarks_frame) < 408:
            # Pad if needed
            landmarks_frame = np.pad(landmarks_frame, (0, max(0, 408 - len(landmarks_frame))), 'constant')
        
        pose = landmarks_frame[:132].reshape(33, 4)[:, :3]  # Only x,y,z
        left_hand = landmarks_frame[132:195].reshape(21, 3)
        right_hand = landmarks_frame[195:258].reshape(21, 3)
        
        features = []
        
        # Hand-to-face distance (use nose as face reference)
        nose = pose[0] if len(pose) > 0 else np.array([0.5, 0.5, 0.0])
        
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
        if len(pose) >= 13:
            left_shoulder = pose[11]
            right_shoulder = pose[12]
            shoulder_midpoint = (left_shoulder + right_shoulder) / 2
        else:
            shoulder_midpoint = np.array([0.5, 0.5, 0.0])
        
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
        
        return np.array(features, dtype=np.float32)


