"""
Video preprocessing pipeline for sign language data
Extracts MediaPipe landmarks and prepares data for training
"""
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
import pickle
from tqdm import tqdm
from typing import Dict, List, Optional
import albumentations as A


class SignLanguagePreprocessor:
    """Preprocess sign language videos using MediaPipe Holistic"""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # MediaPipe Holistic setup
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Augmentation pipeline (for training data)
        self.augmentation = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        ])
    
    def process_video(self, video_path: Path, augment: bool = False) -> Dict:
        """
        Process a video and extract landmarks
        
        Args:
            video_path: Path to video file
            augment: Whether to apply data augmentation
            
        Returns:
            Dictionary with landmarks sequence and metadata
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
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
            'landmarks': np.array(landmarks_sequence, dtype=np.float32),
            'fps': fps,
            'num_frames': frame_count,
            'resolution': [width, height],
            'video_path': str(video_path)
        }
    
    def _extract_and_normalize_landmarks(self, results) -> np.ndarray:
        """
        Extract landmarks and normalize them to fixed-size feature vector
        
        Returns:
            Fixed-size feature vector (408 dimensions)
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
                if idx < len(results.face_landmarks.landmark):
                    lm = results.face_landmarks.landmark[idx]
                    face_features.extend([lm.x, lm.y, lm.z])
                else:
                    face_features.extend([0.0, 0.0, 0.0])
            features.extend(face_features)
        else:
            features.extend([0.0] * 150)
        
        # Total: 132 + 63 + 63 + 150 = 408 features per frame
        return np.array(features, dtype=np.float32)
    
    def process_dataset(self, annotation_file: str, split: str = 'train'):
        """
        Process entire dataset based on annotations
        
        Args:
            annotation_file: Path to JSON annotation file
            split: Dataset split name (train/val/test)
        """
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        processed_data = []
        failed_videos = []
        
        for item in tqdm(annotations, desc=f"Processing {split}"):
            video_path = self.input_dir / item['video_path']
            
            if not video_path.exists():
                print(f"Warning: {video_path} not found")
                failed_videos.append(str(video_path))
                continue
            
            try:
                # Process video
                augment = (split == 'train')  # Only augment training data
                processed = self.process_video(video_path, augment=augment)
                
                # Add annotation info
                processed['label'] = item['sign_label']
                processed['gloss'] = item.get('gloss', item['sign_label'])
                processed['signer_id'] = item.get('signer_id', 'unknown')
                
                processed_data.append(processed)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                failed_videos.append(str(video_path))
        
        # Save processed data
        output_path = self.output_dir / f'{split}_processed.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        print(f"\nProcessed {len(processed_data)} videos for {split}")
        if failed_videos:
            print(f"Failed to process {len(failed_videos)} videos")
        
        return processed_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess sign language videos')
    parser.add_argument('--input-dir', type=str, default='data/raw',
                       help='Input directory with videos')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--annotation-file', type=str, required=True,
                       help='JSON file with video annotations')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='Dataset split')
    
    args = parser.parse_args()
    
    preprocessor = SignLanguagePreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    preprocessor.process_dataset(args.annotation_file, args.split)


