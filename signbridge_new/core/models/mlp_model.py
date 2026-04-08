"""
MLP model for fast sign language recognition
Based on best practices from Sign-Language-Recognition-System project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseSignModel


class SignMLPModel(BaseSignModel):
    """
    MLP model for fast sign language recognition using landmark features
    
    Architecture:
    - Multi-layer perceptron
    - ReLU activations
    - Dropout for regularization
    """
    
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 126,  # 63 per hand (2 hands)
        hidden_dims: list = None,
        dropout: float = 0.2
    ):
        """
        Initialize MLP model
        
        Args:
            num_classes: Number of sign classes
            input_dim: Input feature dimension (126 for 2 hands, 63 for 1 hand)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__(num_classes, input_dim)
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        self.hidden_dims = hidden_dims
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, input_dim) or (batch, seq_len, input_dim)
               If 3D, takes mean over sequence dimension
            
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Handle both 2D and 3D inputs
        if x.dim() == 3:
            # Take mean over sequence dimension
            x = x.mean(dim=1)
        
        return self.model(x)
    
    def get_config(self) -> dict:
        """Get model configuration"""
        return {
            'num_classes': self.num_classes,
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'model_type': 'MLP'
        }


