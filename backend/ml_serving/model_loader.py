"""
Model loading and inference service
Supports both simple CNN and spatial-temporal models
"""
import torch
import torch.nn as nn
from pathlib import Path
import json
import sys
from typing import Optional, Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.simple_cnn import SimpleCNN
from models.sign_recognition.spatial_temporal_model import SignRecognitionModel


class ModelLoader:
    """Load and manage ML models for inference"""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.class_labels = None
        self.model_type = None
        self.label_mapping = None
    
    def load_model(self, checkpoint_path: Optional[Path] = None, 
                   model_type: str = 'auto') -> bool:
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint (None = auto-detect)
            model_type: 'simple_cnn', 'spatial_temporal', or 'auto'
            
        Returns:
            True if successful, False otherwise
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Auto-detect checkpoint if not provided
        if checkpoint_path is None:
            # Try spatial-temporal first, then simple CNN
            st_path = project_root / 'models' / 'checkpoints' / 'sign-recognition-best.ckpt'
            simple_path = project_root / 'models' / 'checkpoints' / 'best_model.pth'
            
            if st_path.exists() and model_type in ['auto', 'spatial_temporal']:
                checkpoint_path = st_path
                model_type = 'spatial_temporal'
            elif simple_path.exists() and model_type in ['auto', 'simple_cnn']:
                checkpoint_path = simple_path
                model_type = 'simple_cnn'
            else:
                print(f"Warning: No model checkpoint found")
                return False
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            return False
        
        try:
            if model_type == 'spatial_temporal' or checkpoint_path.suffix == '.ckpt':
                # PyTorch Lightning checkpoint
                from pytorch_lightning import LightningModule
                
                # Load checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Extract hyperparameters
                if 'hyper_parameters' in checkpoint:
                    hparams = checkpoint['hyper_parameters']
                    num_classes = hparams.get('num_classes', 24)
                else:
                    # Try to infer from state dict
                    num_classes = checkpoint.get('num_classes', 24)
                
                # Create model
                self.model = SignRecognitionModel(
                    input_dim=408,
                    num_classes=num_classes
                )
                
                # Load weights (handle Lightning format)
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    # Remove 'model.' prefix if present
                    state_dict = {k.replace('model.', ''): v 
                                for k, v in state_dict.items()}
                    self.model.load_state_dict(state_dict, strict=False)
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.model_type = 'spatial_temporal'
                
                # Load label mapping
                label_mapping_path = checkpoint_path.parent / 'label_mapping.json'
                if label_mapping_path.exists():
                    with open(label_mapping_path, 'r') as f:
                        self.label_mapping = json.load(f)
                    self.class_labels = list(self.label_mapping['label_to_idx'].keys())
                else:
                    # Default ASL labels
                    self.class_labels = [chr(i) for i in range(ord('A'), ord('Z')+1) 
                                       if chr(i) not in ['J', 'Z']]
                
            else:
                # Simple CNN model
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                num_classes = checkpoint.get('num_classes', 24)
                
                self.model = SimpleCNN(num_classes=num_classes)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model_type = 'simple_cnn'
                
                # ASL labels for simple CNN
                self.class_labels = [chr(i) for i in range(ord('A'), ord('Z')+1) 
                                   if chr(i) not in ['J', 'Z']]
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully! Type: {self.model_type}")
            print(f"Number of classes: {len(self.class_labels)}")
            print(f"Classes: {self.class_labels[:10]}...")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, input_tensor: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None) -> Dict:
        """
        Run inference
        
        Args:
            input_tensor: Input tensor
            lengths: Sequence lengths (for spatial-temporal model)
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        with torch.no_grad():
            if self.model_type == 'spatial_temporal':
                # Spatial-temporal model expects (batch, seq_len, features)
                if len(input_tensor.shape) == 2:
                    # Single frame, add batch and sequence dimensions
                    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
                elif len(input_tensor.shape) == 3:
                    # Already has batch and sequence
                    pass
                else:
                    raise ValueError(f"Unexpected input shape: {input_tensor.shape}")
                
                if lengths is None:
                    lengths = torch.tensor([input_tensor.size(1)], device=self.device)
                
                logits = self.model(input_tensor.to(self.device), lengths)
            else:
                # Simple CNN expects (batch, channels, height, width)
                if len(input_tensor.shape) == 2:
                    # Add batch and channel dimensions
                    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
                elif len(input_tensor.shape) == 3:
                    input_tensor = input_tensor.unsqueeze(0)
                
                logits = self.model(input_tensor.to(self.device))
            
            probabilities = torch.softmax(logits, dim=1)
            
            # Get top prediction
            confidence, predicted_idx = probabilities.max(1)
            predicted_class = self.class_labels[predicted_idx.item()]
            confidence_score = confidence.item()
            
            # Get top 3 predictions
            top3_probs, top3_indices = probabilities.topk(3, dim=1)
            top3_predictions = [
                {
                    "class": self.class_labels[idx.item()],
                    "confidence": prob.item()
                }
                for prob, idx in zip(top3_probs[0], top3_indices[0])
            ]
            
            return {
                'prediction': predicted_class,
                'confidence': confidence_score,
                'top_3': top3_predictions
            }


