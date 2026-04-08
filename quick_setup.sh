#!/bin/bash

echo "========================================"
echo "SignBridge Quick Setup"
echo "========================================"
echo ""

echo "Step 1: Creating .env file..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file"
else
    echo ".env already exists"
fi
echo ""

echo "Step 2: Installing minimal dependencies for Sprint 1..."
echo "This will install: torch, pandas, numpy, tqdm"
echo ""

# Install PyTorch (CPU version for faster install)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pandas numpy tqdm scikit-learn matplotlib

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Train model: python scripts/training/train_simple_model.py --epochs 5"
echo "2. This will take 10-15 minutes"
echo ""
