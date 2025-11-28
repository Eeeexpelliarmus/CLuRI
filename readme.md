# CLuRI: Contrastive Learning Using Resonance Information for Multimodal Intent Detection

Official implementation of the paper "Contrastive Learning Using Resonance Information for Multimodal Intent Detection".

## Repository Status

🔄 **Active Maintenance**: This repository is currently under active development and maintenance. We are continuously working to improve code quality, expand functionality, and address user feedback. Future updates will include (but are not limited to):

- Bug fixes and performance optimizations for core modules (DDR, FCL).
- Support for additional multimodal intent detection datasets (e.g., SIMMC, MultiWOZ-Intent).
- Extended visualization tools (e.g., attention heatmaps for cross-modal resonance).
- Pre-trained model checkpoints for quick inference.
- Tutorials and Colab notebooks for easier reproduction and adaptation.

Stay tuned for updates by starring this repository!

## Overview

CLuRI addresses three core challenges in multimodal intent detection:

1. **Noise in Non-Linguistic Modalities**: Audio/video data often contain irrelevant noise (e.g., background chatter, frame blur) that degrades feature quality.
2. **Cross-Modal Misalignment**: Semantic gaps between text, audio, and video make it hard to fuse their information effectively.
3. **Data Scarcity**: Collecting labeled multimodal intent data is costly, limiting model generalization.

To solve these, CLuRI introduces three key innovations:

1. **Dual Domain Resonance (DDR) Module**: Separates temporal (short-term semantics/long-term dependencies) and spectral (frequency-specific features) information for audio/video, then aligns cross-modal "resonance frequencies" via a Unified Cavity (UC) mechanism.
2. **Fine-Grained Contrastive Learning (FCL)**: Combines global intent representations (from [CLS] tokens) and local semantic constraints (from masked features) to learn more discriminative multimodal embeddings.
3. **ChatGPT-Based Data Augmentation**: Generates diverse, intent-consistent text expressions to expand training data at low cost, without re-collecting audio/video.

Experimental results on the MIntRec and MELD-DA datasets show CLuRI outperforms state-of-the-art baselines across accuracy, F1-score, precision, and recall.

## Key Features

- **Modular Architecture**: Easy to modify or ablate individual components (e.g., disable DDR-SB for ablation studies).

- State-of-the-Art Feature Extractors

  : Integrates pre-trained models for each modality:

  - Text: BERT (`bert-base-uncased`).
  - Video: Swin Transformer (pre-trained on ImageNet).
  - Audio: Wav2Vec 2.0 (`wav2vec2-base-960h`).

- **Reproducibility**: Includes full training/evaluation pipelines, preprocessing scripts, and hyperparameter configurations matching the paper.

- **Comprehensive Tools**: Built-in visualization for feature distributions, model confidence, and cross-modal alignment.

## Environment Setup

### Prerequisites

- Python 3.8–3.10 (compatibility tested).
- PyTorch 1.10+ (with CUDA support recommended for training speed).
- Hugging Face Transformers 4.18+ (for BERT/Wav2Vec 2.0).
- Audio/Video Processing: `librosa` (0.9+), `opencv-python` (4.5+), `ffmpeg` (for frame extraction).
- Utilities: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `tqdm`.
- OpenAI API Key (required only for ChatGPT data augmentation).

### Installation

```
# Clone the repository
git clone https://github.com/your-username/CLuRI.git
cd CLuRI

# Install dependencies (use pip3 if Python 2 is default)
pip install -r requirements.txt

# Optional: Install ffmpeg for video processing (Linux example)
sudo apt update && sudo apt install ffmpeg
```

## License

This project is licensed under the **MIT License**—see the [LICENSE](https://www.doubao.com/chat/LICENSE) file for details.