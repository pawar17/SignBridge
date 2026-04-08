"""
SignBridge Landmark Extractor
Extracts MediaPipe landmarks from video files or images
"""
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
from tqdm import tqdm
from typing import Optional, Dict, List
import argparse


class LandmarkExtractor:
    """Extract pose, hand, and face landmarks from videos/images"""

    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False
    ):
        """
        Initialize MediaPipe Holistic

        Args:
            model_complexity: Model complexity (0, 1, or 2)
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
            static_image_mode: Whether to treat input as static images
        """
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def extract_from_video(self, video_path: str) -> Optional[np.ndarray]:
        """
        Extract landmarks from a video file

        Args:
            video_path: Path to video file

        Returns:
            Array of shape (num_frames, 408) containing landmarks
            Returns None if extraction fails
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None

        landmarks_sequence = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB (MediaPipe uses RGB)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = self.holistic.process(image_rgb)

            # Extract and normalize landmarks
            frame_landmarks = self._extract_landmarks(results)
            landmarks_sequence.append(frame_landmarks)

            frame_idx += 1

        cap.release()

        if len(landmarks_sequence) == 0:
            return None

        return np.array(landmarks_sequence, dtype=np.float32)

    def extract_from_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract landmarks from a single image

        Args:
            image_path: Path to image file

        Returns:
            Array of shape (408,) containing landmarks
            Returns None if extraction fails
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not open image {image_path}")
            return None

        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self.holistic.process(image_rgb)

        # Extract landmarks
        landmarks = self._extract_landmarks(results)

        return landmarks

    def _extract_landmarks(self, results) -> np.ndarray:
        """
        Extract all landmarks from MediaPipe results

        Returns:
            Array of 408 features:
            - Pose: 132 (33 points × 4 features: x, y, z, visibility)
            - Left hand: 63 (21 points × 3 features: x, y, z)
            - Right hand: 63 (21 points × 3 features: x, y, z)
            - Face: 150 (50 key points × 3 features: x, y, z)
        """
        features = []

        # Pose landmarks (33 points × 4 features = 132)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            features.extend([0.0] * 132)

        # Left hand (21 points × 3 features = 63)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * 63)

        # Right hand (21 points × 3 features = 63)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * 63)

        # Face landmarks (50 key points × 3 features = 150)
        # Select key facial points for expressions
        if results.face_landmarks:
            key_indices = [
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
            for idx in key_indices:
                lm = results.face_landmarks.landmark[idx]
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * 150)

        return np.array(features, dtype=np.float32)

    def process_dataset(
        self,
        input_dir: str,
        output_dir: str,
        file_extension: str = ".mp4"
    ) -> None:
        """
        Process entire dataset directory

        Args:
            input_dir: Input directory containing videos/images
            output_dir: Output directory for landmarks
            file_extension: File extension to process (.mp4, .jpg, etc.)
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all files
        files = list(input_path.rglob(f"*{file_extension}"))
        print(f"Found {len(files)} files to process")

        # Process each file
        for file_path in tqdm(files, desc="Processing files"):
            try:
                # Extract landmarks
                if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
                    landmarks = self.extract_from_video(str(file_path))
                else:
                    landmarks = self.extract_from_image(str(file_path))

                if landmarks is not None:
                    # Create output path with same structure
                    relative_path = file_path.relative_to(input_path)
                    output_file = output_path / relative_path.with_suffix('.npy')
                    output_file.parent.mkdir(parents=True, exist_ok=True)

                    # Save landmarks
                    np.save(output_file, landmarks)
                else:
                    print(f"Failed to extract landmarks from {file_path}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        print(f"\nProcessing complete! Landmarks saved to {output_dir}")

    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'holistic'):
            self.holistic.close()


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe landmarks from videos/images"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing videos/images"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for landmarks"
    )
    parser.add_argument(
        "--extension",
        type=str,
        default=".mp4",
        help="File extension to process (default: .mp4)"
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="MediaPipe model complexity (default: 2)"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence (default: 0.5)"
    )

    args = parser.parse_args()

    # Create extractor
    extractor = LandmarkExtractor(
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_confidence,
        min_tracking_confidence=args.min_confidence
    )

    # Process dataset
    extractor.process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        file_extension=args.extension
    )


if __name__ == "__main__":
    main()
