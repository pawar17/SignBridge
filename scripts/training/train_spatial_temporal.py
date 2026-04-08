"""
Training script for spatial-temporal sign recognition model
Uses PyTorch Lightning for training infrastructure
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.sign_recognition.spatial_temporal_model import SignRecognitionModel


class SignLanguageDataset(Dataset):
    """Dataset for sign language recognition"""
    
    def __init__(self, data_file: str, max_seq_len: int = 150):
        """
        Args:
            data_file: Path to pickle file with processed data
            max_seq_len: Maximum sequence length (frames)
        """
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        
        self.max_seq_len = max_seq_len
        
        # Build label mapping
        unique_labels = set(item['label'] for item in self.data)
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        print(f"Loaded {len(self.data)} samples")
        print(f"Number of classes: {len(self.label_to_idx)}")
        print(f"Classes: {sorted(self.label_to_idx.keys())[:10]}...")  # Show first 10
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get landmarks
        landmarks = item['landmarks']  # Shape: (num_frames, 408)
        
        # Truncate or pad
        if len(landmarks) > self.max_seq_len:
            landmarks = landmarks[:self.max_seq_len]
            seq_len = self.max_seq_len
        else:
            pad_len = self.max_seq_len - len(landmarks)
            landmarks = np.pad(landmarks, ((0, pad_len), (0, 0)), 'constant')
            seq_len = len(item['landmarks'])
        
        # Get label
        label = self.label_to_idx[item['label']]
        
        return {
            'landmarks': torch.FloatTensor(landmarks),
            'label': torch.LongTensor([label]),
            'seq_len': torch.LongTensor([seq_len])
        }


class SignRecognitionLightning(pl.LightningModule):
    """PyTorch Lightning module for sign recognition"""
    
    def __init__(self, num_classes: int, learning_rate: float = 1e-4,
                 input_dim: int = 408, spatial_dim: int = 256,
                 temporal_dim: int = 256, num_transformer_layers: int = 4):
        super().__init__()
        
        self.model = SignRecognitionModel(
            input_dim=input_dim,
            spatial_dim=spatial_dim,
            temporal_dim=temporal_dim,
            num_classes=num_classes,
            num_transformer_layers=num_transformer_layers
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        
        self.save_hyperparameters()
    
    def forward(self, x, lengths):
        return self.model(x, lengths)
    
    def training_step(self, batch, batch_idx):
        landmarks = batch['landmarks']
        labels = batch['label'].squeeze()
        lengths = batch['seq_len'].squeeze()
        
        logits = self(landmarks, lengths)
        loss = self.criterion(logits, labels)
        
        # Accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        landmarks = batch['landmarks']
        labels = batch['label'].squeeze()
        lengths = batch['seq_len'].squeeze()
        
        logits = self(landmarks, lengths)
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


def train_model(args):
    """Main training function"""
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = SignLanguageDataset(args.train_data, max_seq_len=args.max_seq_len)
    val_dataset = SignLanguageDataset(args.val_data, max_seq_len=args.max_seq_len)
    
    # Ensure label mappings match
    if train_dataset.label_to_idx != val_dataset.label_to_idx:
        print("Warning: Label mappings differ between train and val sets!")
        # Use train set's mapping
        val_dataset.label_to_idx = train_dataset.label_to_idx
        val_dataset.idx_to_label = train_dataset.idx_to_label
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Model
    num_classes = len(train_dataset.label_to_idx)
    print(f"\nCreating model with {num_classes} classes...")
    
    model = SignRecognitionLightning(
        num_classes=num_classes,
        learning_rate=args.learning_rate,
        input_dim=args.input_dim,
        spatial_dim=args.spatial_dim,
        temporal_dim=args.temporal_dim,
        num_transformer_layers=args.num_transformer_layers
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='sign-recognition-{epoch:02d}-{val_acc:.2f}',
        monitor='val_acc',
        mode='max',
        save_top_k=3,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        mode='min',
        verbose=True
    )
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name='sign_recognition'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator='auto',  # Auto-detect GPU/CPU
        devices=1,
        precision=16 if args.mixed_precision else 32,
        gradient_clip_val=1.0,
        log_every_n_steps=10
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)
    
    print(f"\nTraining complete! Best model saved to: {checkpoint_callback.best_model_path}")
    
    # Save label mappings
    label_mapping_path = Path(args.output_dir) / 'label_mapping.json'
    import json
    with open(label_mapping_path, 'w') as f:
        json.dump({
            'label_to_idx': train_dataset.label_to_idx,
            'idx_to_label': train_dataset.idx_to_label
        }, f, indent=2)
    print(f"Label mappings saved to: {label_mapping_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train spatial-temporal sign recognition model')
    
    # Data arguments
    parser.add_argument('--train-data', type=str, required=True,
                       help='Path to training data pickle file')
    parser.add_argument('--val-data', type=str, required=True,
                       help='Path to validation data pickle file')
    parser.add_argument('--max-seq-len', type=int, default=150,
                       help='Maximum sequence length')
    
    # Model arguments
    parser.add_argument('--input-dim', type=int, default=408,
                       help='Input feature dimension')
    parser.add_argument('--spatial-dim', type=int, default=256,
                       help='Spatial encoder hidden dimension')
    parser.add_argument('--temporal-dim', type=int, default=256,
                       help='Temporal encoder hidden dimension')
    parser.add_argument('--num-transformer-layers', type=int, default=4,
                       help='Number of transformer layers')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='models/checkpoints',
                       help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    train_model(args)


