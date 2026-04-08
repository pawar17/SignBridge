"""
LSTM model for temporal sign language recognition
Based on best practices from Realtime-Sign-Language-Detection project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseSignModel


class SignLSTMModel(BaseSignModel):
    """
    LSTM model for sign language recognition using temporal sequences
    
    Architecture:
    - 3-layer LSTM for temporal modeling
    - Dense layers for classification
    - Dropout for regularization
    """
    
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 1662,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        """
        Initialize LSTM model
        
        Args:
            num_classes: Number of sign classes
            input_dim: Input feature dimension (1662 for MediaPipe Holistic)
            hidden_dim: Hidden dimension for LSTM layers
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__(num_classes, input_dim)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            return_sequences=True
        )
        self.lstm2 = nn.LSTM(
            hidden_dim,
            hidden_dim * 2,
            batch_first=True,
            return_sequences=True
        )
        self.lstm3 = nn.LSTM(
            hidden_dim * 2,
            hidden_dim,
            batch_first=True,
            return_sequences=False
        )
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        
        # Classification
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def get_config(self) -> dict:
        """Get model configuration"""
        return {
            'num_classes': self.num_classes,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'model_type': 'LSTM'
        }


