"""
Sign recording tool for creating custom datasets
Records videos with MediaPipe landmark extraction
"""
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import mediapipe as mp


class SignRecorder:
    """Record sign language videos with landmark extraction"""
    
    def __init__(self, output_dir: str = 'data/raw/custom'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # MediaPipe Holistic setup
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def record_sign(self, sign_label: str, signer_id: str = 'signer_001', 
                    session_id: str = 'session_001'):
        """
        Record a single sign with metadata
        
        Args:
            sign_label: Label for the sign (e.g., 'HELLO')
            signer_id: ID of the signer
            session_id: Session identifier
            
        Returns:
            Metadata dictionary or None if cancelled
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return None
        
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
        
        print(f"\nRecording '{sign_label}'")
        print("Controls:")
        print("  'r' - Start/Resume recording")
        print("  's' - Stop recording and save")
        print("  'q' - Quit without saving")
        
        recording = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process with MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image_rgb)
            
            # Draw landmarks on frame
            self._draw_landmarks(frame, results)
            
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
            
            cv2.putText(display_frame, f"Sign: {sign_label}", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Frames: {len(frames)}", (50, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Sign Recording', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                recording = True
                print("Started recording...")
            elif key == ord('s'):
                recording = False
                if len(frames) > 0:
                    print(f"Stopped recording. Saved {len(frames)} frames.")
                    break
                else:
                    print("No frames recorded. Press 'r' to start recording.")
            elif key == ord('q'):
                print("Quit without saving.")
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                return None
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        if len(frames) == 0:
            print("No frames recorded. Recording cancelled.")
            return None
        
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
        
        # Face landmarks (key points for expressions)
        if results.face_landmarks:
            key_indices = list(range(0, 17)) + list(range(33, 133)) + list(range(362, 400))
            landmarks['face'] = [
                [results.face_landmarks.landmark[i].x,
                 results.face_landmarks.landmark[i].y,
                 results.face_landmarks.landmark[i].z]
                for i in key_indices if i < len(results.face_landmarks.landmark)
            ]
        
        return landmarks
    
    def _draw_landmarks(self, frame, results):
        """Draw MediaPipe landmarks on frame"""
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_holistic.draw_landmarks(
                frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        
        # Draw hand landmarks
        if results.left_hand_landmarks:
            self.mp_holistic.draw_landmarks(
                frame, results.left_hand_landmarks, 
                self.mp_holistic.HAND_CONNECTIONS)
        
        if results.right_hand_landmarks:
            self.mp_holistic.draw_landmarks(
                frame, results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS)
        
        # Draw face landmarks (optional, can be commented for performance)
        # if results.face_landmarks:
        #     self.mp_holistic.draw_landmarks(
        #         frame, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Record sign language videos')
    parser.add_argument('--sign', type=str, required=True,
                       help='Sign label to record')
    parser.add_argument('--signer-id', type=str, default='signer_001',
                       help='Signer identifier')
    parser.add_argument('--session-id', type=str, default='session_001',
                       help='Session identifier')
    parser.add_argument('--output-dir', type=str, default='data/raw/custom',
                       help='Output directory')
    
    args = parser.parse_args()
    
    recorder = SignRecorder(output_dir=args.output_dir)
    recorder.record_sign(
        sign_label=args.sign,
        signer_id=args.signer_id,
        session_id=args.session_id
    )


