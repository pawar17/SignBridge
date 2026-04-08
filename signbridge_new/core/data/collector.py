"""
Interactive data collection tool for sign language recognition
Based on Realtime-Sign-Language-Detection project
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
from collections import deque
import json
from datetime import datetime

from ..feature_extractor import FeatureExtractor, FeatureMode


class DataCollector:
    """
    Interactive data collection tool for sign language recognition
    
    Features:
    - Real-time webcam collection
    - Sequence-based data capture
    - Automatic landmark extraction
    - Organized data storage
    """
    
    def __init__(
        self,
        output_dir: str = "data/raw",
        mode: FeatureMode = FeatureMode.HOLISTIC,
        sequence_length: int = 30,
        fps: int = 30
    ):
        """
        Initialize data collector
        
        Args:
            output_dir: Output directory for collected data
            mode: Feature extraction mode
            sequence_length: Number of frames per sequence
            fps: Target FPS for collection
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_extractor = FeatureExtractor(mode=mode)
        self.sequence_length = sequence_length
        self.fps = fps
        
        self.cap = None
        self.current_sign = None
        self.current_sequence = None
        self.sequence_count = 0
    
    def collect_sequence(
        self,
        sign_label: str,
        num_sequences: int = 30,
        sequences_per_sign: int = 30
    ) -> bool:
        """
        Collect sequences for a sign
        
        Args:
            sign_label: Label for the sign
            num_sequences: Number of sequences to collect
            sequences_per_sign: Total sequences per sign (for progress)
            
        Returns:
            True if collection completed, False if cancelled
        """
        self.current_sign = sign_label
        sign_dir = self.output_dir / sign_label
        sign_dir.mkdir(exist_ok=True)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        print(f"\n{'='*50}")
        print(f"Collecting data for: {sign_label}")
        print(f"Sequences to collect: {num_sequences}")
        print(f"Frames per sequence: {self.sequence_length}")
        print(f"{'='*50}\n")
        
        for seq_idx in range(num_sequences):
            print(f"Sequence {seq_idx + 1}/{num_sequences}")
            print("Press SPACE to start recording, 'q' to quit")
            
            # Wait for space key
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                cv2.putText(
                    frame,
                    f"Sign: {sign_label} | Seq: {seq_idx + 1}/{num_sequences}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    frame,
                    "Press SPACE to start, 'q' to quit",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                cv2.imshow("Data Collection", frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):
                    break
                elif key == ord('q'):
                    self.cap.release()
                    cv2.destroyAllWindows()
                    return False
            
            # Collect sequence
            sequence_dir = sign_dir / str(seq_idx)
            sequence_dir.mkdir(exist_ok=True)
            
            landmarks_sequence = []
            
            for frame_idx in range(self.sequence_length):
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Extract landmarks
                landmarks = self.feature_extractor.extract_landmarks(frame)
                landmarks_sequence.append(landmarks)
                
                # Save frame landmarks
                np.save(sequence_dir / f"{frame_idx}.npy", landmarks)
                
                # Display
                cv2.putText(
                    frame,
                    f"Recording... Frame: {frame_idx + 1}/{self.sequence_length}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
                cv2.imshow("Data Collection", frame)
                cv2.waitKey(int(1000 / self.fps))
            
            # Save sequence metadata
            metadata = {
                'sign_label': sign_label,
                'sequence_idx': seq_idx,
                'num_frames': len(landmarks_sequence),
                'timestamp': datetime.now().isoformat(),
                'feature_dim': self.feature_extractor.feature_dim
            }
            
            with open(sequence_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ Sequence {seq_idx + 1} saved\n")
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"✓ Completed collection for {sign_label}")
        return True
    
    def collect_batch(self, signs: List[str], sequences_per_sign: int = 30):
        """
        Collect data for multiple signs
        
        Args:
            signs: List of sign labels
            sequences_per_sign: Number of sequences per sign
        """
        print(f"\n{'='*50}")
        print(f"Batch Data Collection")
        print(f"Signs: {', '.join(signs)}")
        print(f"Sequences per sign: {sequences_per_sign}")
        print(f"{'='*50}\n")
        
        for sign in signs:
            success = self.collect_sequence(sign, sequences_per_sign)
            if not success:
                print("\nCollection cancelled by user")
                break
        
        print("\n✓ Batch collection completed")
    
    def get_collected_signs(self) -> List[str]:
        """Get list of collected signs"""
        if not self.output_dir.exists():
            return []
        
        return [d.name for d in self.output_dir.iterdir() if d.is_dir()]


