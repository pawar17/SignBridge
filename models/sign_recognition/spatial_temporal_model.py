"""
Spatial-Temporal Model for Sign Language Recognition
Combines CNN for spatial features and Transformer for temporal modeling
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class SpatialEncoder(nn.Module):
    """Encodes spatial features from a single frame"""
    def __init__(self, input_dim: int = 408, hidden_dim: int = 256):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(0.3)
        
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        batch, seq_len, input_dim = x.shape
        
        # Reshape for batch norm
        x = x.view(batch * seq_len, input_dim)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        
        # Reshape back
        x = x.view(batch, seq_len, -1)
        
        return x


class TemporalEncoder(nn.Module):
    """Encodes temporal dynamics using Transformer"""
    def __init__(self, input_dim: int = 256, num_layers: int = 4, 
                 nhead: int = 8, dim_feedforward: int = 1024):
        super().__init__()
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(input_dim)
    
    def forward(self, x, src_key_padding_mask: Optional[torch.Tensor] = None):
        # x shape: (batch, seq_len, input_dim)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return x


class SignRecognitionModel(nn.Module):
    """
    Complete model: Spatial + Temporal encoding + Classification
    """
    def __init__(self, 
                 input_dim: int = 408,
                 spatial_dim: int = 256,
                 temporal_dim: int = 256,
                 num_classes: int = 1000,
                 num_transformer_layers: int = 4):
        super().__init__()
        
        self.spatial_encoder = SpatialEncoder(input_dim, spatial_dim)
        self.temporal_encoder = TemporalEncoder(
            spatial_dim, 
            num_layers=num_transformer_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(temporal_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x, lengths: Optional[torch.Tensor] = None):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            lengths: Actual sequence lengths for masking (batch,)
            
        Returns:
            Logits (batch, num_classes)
        """
        # Spatial encoding
        spatial_features = self.spatial_encoder(x)
        
        # Create padding mask if lengths provided
        if lengths is not None:
            batch_size, max_len = x.size(0), x.size(1)
            mask = torch.arange(max_len).expand(batch_size, max_len).to(x.device)
            mask = mask >= lengths.unsqueeze(1)
        else:
            mask = None
        
        # Temporal encoding
        temporal_features = self.temporal_encoder(spatial_features, mask)
        
        # Global pooling (mean over time, ignoring padding)
        if mask is not None:
            temporal_features = temporal_features.masked_fill(mask.unsqueeze(-1), 0)
            pooled = temporal_features.sum(dim=1) / lengths.unsqueeze(1).float()
        else:
            pooled = temporal_features.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits


class Seq2SeqTranslator(nn.Module):
    """
    Translates sign gloss sequence to natural language
    """
    def __init__(self, 
                 vocab_size_src: int,
                 vocab_size_tgt: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6):
        super().__init__()
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        
        self.src_embedding = nn.Embedding(vocab_size_src, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size_tgt, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size_tgt)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embedding + positional encoding
        src = self.pos_encoder(self.src_embedding(src))
        tgt = self.pos_encoder(self.tgt_embedding(tgt))
        
        # Transformer
        output = self.transformer(
            src, tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
        
        # Project to vocabulary
        output = self.fc_out(output)
        
        return output


