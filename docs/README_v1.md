# SignBridge: Bidirectional Sign Language Translation & Learning Platform

![Project Status](https://img.shields.io/badge/status-in--development-yellow)
![License](https://img.shields.io/badge/license-MIT-blue)

## Overview

SignBridge is an AI-powered platform that facilitates bidirectional communication between sign language users and non-signers through real-time translation, while providing an adaptive learning environment for sign language acquisition.

### Key Features

- **Bidirectional Translation**: Real-time sign language ↔ text/speech translation
- **Multi-Language Support**: ASL, ISL, BSL, JSL, LSF with dialect recognition
- **Adaptive Learning**: Personalized sign language learning curriculum
- **Context-Aware Processing**: Understands grammar differences between sign and spoken languages
- **Privacy-First**: On-device processing options for sensitive environments
- **Accessible Design**: WCAG 2.2 AAA compliance

## Project Structure

```
SignBridge/
├── docs/                 # Documentation
├── data/                 # Datasets and processed data
├── models/               # ML models and training code
├── backend/              # FastAPI backend
├── frontend/             # React frontend
├── notebooks/            # Jupyter notebooks
├── scripts/              # Utility scripts
├── tests/                # Test files
└── docker/               # Docker configurations
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- CUDA-compatible GPU (recommended for training)
- Docker & Docker Compose (for deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/SignBridge.git
   cd SignBridge
   ```

2. **Set up Python environment**
   ```bash
   conda create -n signbridge python=3.10
   conda activate signbridge
   pip install -r requirements.txt
   ```

3. **Set up frontend**
   ```bash
   cd frontend
   npm install
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Running the Application

**Development Mode:**

```bash
# Backend
cd backend
uvicorn api.main:app --reload

# Frontend
cd frontend
npm start
```

**Production Mode (Docker):**

```bash
docker-compose up -d
```

## Datasets

SignBridge uses multiple sign language datasets:

- **ASL MNIST**: 27,456 training + 7,173 test samples
- **Custom ASL Dataset**: 36 classes (0-9, a-z)
- **Indian Sign Language**: Custom collected dataset
- **German Sign Language**: CSV dataset with alphabet reference

See `data/README.md` for detailed dataset information.

## Development

### Data Preprocessing

```bash
python scripts/data_preprocessing/landmark_extractor.py
python scripts/data_preprocessing/dataset_builder.py
```

### Model Training

```bash
python scripts/training/train_model.py --config configs/train_config.yaml
```

### Running Tests

```bash
pytest tests/
```

## Documentation

- [Product Requirements Document](docs/PRD.md)
- [Implementation Guide](docs/implementation_guide.md)
- [Claude Code Guide](docs/claude_code_guide.md)
- [API Documentation](docs/api/)
- [Architecture](docs/architecture/)

## Technology Stack

**Frontend:**
- React.js with TypeScript
- Three.js for 3D avatar rendering
- TensorFlow.js for on-device inference
- Material-UI components

**Backend:**
- FastAPI (Python)
- PyTorch for ML models
- MediaPipe for pose estimation
- PostgreSQL database
- Redis for caching

**ML/AI:**
- Custom Transformer + LSTM hybrid for sign recognition
- Seq2Seq model for text-to-sign generation
- MediaPipe Holistic for landmark extraction

**Infrastructure:**
- Docker & Kubernetes
- AWS/GCP for cloud deployment
- GitHub Actions for CI/CD

## Roadmap

### Phase 1: Foundation (Months 1-3) ✓
- [x] Project setup and structure
- [x] Dataset collection and organization
- [ ] Basic sign recognition model
- [ ] MVP translation interface

### Phase 2: Enhancement (Months 4-6)
- [ ] Bidirectional translation
- [ ] Multi-language support (ASL + ISL)
- [ ] Adaptive learning system
- [ ] Mobile-responsive UI

### Phase 3: Scale & Research (Months 7-12)
- [ ] Multi-language support (BSL, JSL, LSF)
- [ ] Production deployment
- [ ] Research paper submissions
- [ ] Community partnerships

## Contributing

We welcome contributions from the community! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Research & Publications

SignBridge is designed with research goals in mind:

- **Few-shot signer adaptation** for personalization
- **Cross-lingual sign language transfer** learning
- **Grammar-aware translation** preserving linguistic structure
- **Cognitive load assessment** for accessible learning

## Ethics & Privacy

- **Community Partnership**: Development guided by DHH advisory board
- **Privacy-First**: On-device processing options available
- **Transparent AI**: Clear communication about capabilities and limitations
- **Accessibility**: Tool itself fully usable by DHH individuals

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MediaPipe team for pose estimation tools
- Sign language datasets: WLASL, How2Sign, MS-ASL, INCLUDE
- DHH community for guidance and feedback
- Open-source contributors

## Contact

- **Project Lead**: [Your Name]
- **Email**: your.email@example.com
- **Website**: https://signbridge.example.com
- **Issues**: [GitHub Issues](https://github.com/yourusername/SignBridge/issues)

## Citation

If you use SignBridge in your research, please cite:

```bibtex
@software{signbridge2024,
  title={SignBridge: Bidirectional Sign Language Translation and Learning Platform},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/SignBridge}
}
```

---

**Made with ❤️ for the Deaf and Hard of Hearing community**
