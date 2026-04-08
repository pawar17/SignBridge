"""
Enhanced training script for ASL MNIST
Includes data augmentation for better accuracy
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.simple_cnn import SimpleCNN


class ASLMNISTDataset(Dataset):
    """Dataset for ASL MNIST with data augmentation"""

    def __init__(self, csv_path, augment=False):
        """
        Args:
            csv_path: Path to CSV file
            augment: Whether to apply data augmentation
        """
        self.data = pd.read_csv(csv_path)
        self.augment = augment

        # Separate labels and pixels
        original_labels = self.data['label'].values
        self.pixels = self.data.iloc[:, 1:].values  # All columns except first

        # Create label mapping: ASL MNIST has labels 0-8, 10-24 (missing 9=J, 25=Z)
        # Map to contiguous 0-23
        unique_labels = sorted(np.unique(original_labels))
        self.label_to_idx = {old_label: idx for idx, old_label in enumerate(unique_labels)}
        self.labels = np.array([self.label_to_idx[label] for label in original_labels])

        # Normalize to [0, 1]
        self.pixels = self.pixels.astype(np.float32) / 255.0

        # Data augmentation transforms (applied during training)
        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomAffine(
                    degrees=15,  # Rotate up to 15 degrees
                    translate=(0.1, 0.1),  # Translate up to 10%
                    scale=(0.9, 1.1),  # Scale between 90% and 110%
                    shear=5  # Shear up to 5 degrees
                ),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                # Random brightness/contrast
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        else:
            self.transform = None

        print(f"Loaded {len(self.labels)} samples from {csv_path}")
        print(f"Original classes: {unique_labels}")
        print(f"Mapped to: {np.unique(self.labels)}")
        print(f"Augmentation: {'enabled' if augment else 'disabled'}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get label
        label = self.labels[idx]

        # Get pixels and reshape to 28x28
        image = self.pixels[idx].reshape(28, 28)

        # Add channel dimension: (28, 28) -> (1, 28, 28)
        image = np.expand_dims(image, axis=0)

        # Convert to tensor
        image = torch.from_numpy(image).float()
        label = torch.tensor(label, dtype=torch.long)

        # Apply augmentation if enabled
        if self.transform:
            image = self.transform(image)

        return image, label


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total

    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(description='Train ASL MNIST model')
    parser.add_argument('--train-csv', type=str,
                       default='data/raw/asl/mnist/train/sign_mnist_train.csv',
                       help='Path to training CSV')
    parser.add_argument('--test-csv', type=str,
                       default='data/raw/asl/mnist/test/sign_mnist_test.csv',
                       help='Path to test CSV')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='models/checkpoints',
                       help='Output directory for checkpoints')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = ASLMNISTDataset(args.train_csv, augment=True)  # Enable augmentation for training
    test_dataset = ASLMNISTDataset(args.test_csv, augment=False)  # No augmentation for testing

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for Windows compatibility
        pin_memory=True if device.type == 'cuda' else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Create model
    print("\nCreating model...")
    num_classes = len(np.unique(train_dataset.labels))
    print(f"Number of classes: {num_classes}")

    model = SimpleCNN(num_classes=num_classes)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)

    best_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(
            model, test_loader, criterion, device
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'num_classes': num_classes,
                'history': history
            }

            checkpoint_path = output_dir / 'best_model.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"  [*] Saved best model (acc: {val_acc:.2f}%)")

    # Save final model
    final_checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'num_classes': num_classes,
        'history': history
    }
    final_path = output_dir / 'final_model.pth'
    torch.save(final_checkpoint, final_path)

    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Models saved to: {output_dir}")
    print(f"  - Best model: {checkpoint_path}")
    print(f"  - Final model: {final_path}")
    print(f"  - History: {history_path}")


if __name__ == '__main__':
    main()
