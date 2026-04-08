"""
Base model interface for sign language recognition
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseSignModel(nn.Module, ABC):
    """Base class for all sign language recognition models"""
    
    def __init__(self, num_classes: int, input_dim: int = 1662):
        """
        Initialize base model
        
        Args:
            num_classes: Number of sign classes
            input_dim: Input feature dimension (default: 1662 for MediaPipe Holistic)
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        pass
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make prediction (with softmax)
        
        Args:
            x: Input tensor
            
        Returns:
            Prediction probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
        return probs
    
    def predict_class(self, x: torch.Tensor) -> int:
        """
        Predict class index
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class index
        """
        probs = self.predict(x)
        return torch.argmax(probs, dim=-1).item()
    
    def save(self, path: str):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'input_dim': self.input_dim,
            'model_type': self.__class__.__name__
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None):
        """
        Load model from file
        
        Args:
            path: Path to model file
            device: Device to load model on
            
        Returns:
            Loaded model instance
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            num_classes=checkpoint['num_classes'],
            input_dim=checkpoint['input_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model


