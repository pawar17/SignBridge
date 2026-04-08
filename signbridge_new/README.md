# SignBridge - Sign Language Recognition Platform

A comprehensive, production-ready sign language recognition system built from scratch using best practices from multiple research projects.

## Features

- 🎯 **Real-time Sign Recognition** - Live detection using webcam
- 🧠 **Multiple Model Architectures** - LSTM for sequences, MLP for fast inference
- 📚 **Learning Mode** - Interactive learning with feedback
- 📖 **Phrase Dictionary** - Large collection of sign language phrases
- 🔧 **Modular Design** - Clean, extensible architecture
- ⚡ **High Performance** - Optimized for real-time inference

## Architecture

- **Backend**: FastAPI (Python)
- **Frontend**: React
- **ML Framework**: PyTorch
- **Feature Extraction**: MediaPipe Holistic/Hands
- **Models**: LSTM (temporal) + MLP (fast inference)

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
```

### Training

```bash
# Collect data
python scripts/collect_data.py

# Train model
python scripts/train.py --model lstm --epochs 50
```

### Running

```bash
# Start backend
uvicorn backend.main:app --reload

# Start frontend
cd frontend
npm start
```

## Project Structure

```
signbridge/
├── core/              # Core ML components
├── backend/           # FastAPI backend
├── frontend/          # React frontend
├── data/              # Data storage
├── scripts/           # Utility scripts
└── tests/             # Tests
```

## Documentation

- [Architecture Design](ARCHITECTURE_DESIGN.md)
- [Project Evaluation](PROJECT_EVALUATION.md)
- [API Documentation](docs/API.md)

## License

MIT


