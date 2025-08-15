# DualWaveNet: Multimodal Vision Transformer for Depression Classification

**DualWaveNet** is a multimodal deep learning architecture that leverages **Vision Transformers (ViTs)** with **Token Merging (ToMe)** for binary depression classification. It processes **speech-derived spectrograms** and **tempograms** as two separate visual channels, combining them for robust multimodal feature learning. This project uses the **EATD-Corpus**, a Mandarin speech dataset, to detect depression from audio recordings.

## Overview

DualWaveNet is designed to classify individuals as depressed or non-depressed based on speech features. It employs a **dual-branch Vision Transformer** architecture, where:
- **Spectrograms** capture frequency-time representations of speech.
- **Tempograms** capture tempo-based features.
- Features from both branches are fused for binary classification (`0`: non-depressed, `1`: depressed).

The model uses **Token Merging (ToMe)** to reduce computational complexity while maintaining high accuracy, achieving expected improvements over baseline models like CNN (66%) and Vanilla ViT (83%).

## Dataset: EATD-Corpus

- **Participants**: 162 student volunteers (30 depressed, 132 non-depressed).
- **Labels**: Derived from the **Self-Rating Depression Scale (SDS)**.
- **Total Duration**: ~2.26 hours of response audio.
- **Input Format**:
  - **Spectrograms**: Frequency-time representations of speech.
  - **Tempograms**: Tempo-based features.
- **Classes**:
  - `0`: Non-depressed
  - `1`: Depressed

## Experimental Results

| Model                  | Accuracy                   |
|------------------------|----------------------------|
| CNN                    | 66%                        |
| Vanilla ViT            | 83%                        |
| **DualWaveNet (ToMe)** | **↑ Expected Improvement** |

## Features

- **Dual-channel ViT architecture**: Processes spectrograms and tempograms separately.
- **Token Merging (ToMe)**: Reduces computational cost without significant accuracy loss.
- **Early Stopping**: Prevents overfitting during training.
- **Precision-Recall Threshold Tuning**: Optimizes F1-score for imbalanced datasets.
- **Visualization**: Includes confusion matrices and performance plots for analysis.

## Installation

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support (optional, for GPU acceleration)
- Git


## Model Architecture

**DualChannelModel**:
- **Branch 1**: Vision Transformer (ViT) with ToMe for spectrograms.
- **Branch 2**: Vision Transformer (ViT) with ToMe for tempograms.
- **Fusion**: Concatenates embeddings from both branches, followed by a linear classifier and softmax for binary classification.

## Usage

1. **Prepare the Dataset**:
   - Download or preprocess the **EATD-Corpus** to generate spectrograms and tempograms.
   - Ensure data is formatted as described in the dataset section.

2. **Train the Model**:
   ```bash
   python train.py --data_path /path/to/eatd_corpus --output_dir /path/to/save
   ```

3. **Evaluate the Model**:
   ```bash
   python evaluate.py --model_path /path/to/trained_model --data_path /path/to/eatd_corpus
   ```

4. **Visualize Results**:
   - Confusion matrices and performance plots are generated automatically during evaluation.

## Future Work

- Test on the **DAIC-WOZ** dataset (~15 hours, multimodal) for generalizability.
- Incorporate **transcript-based NLP features** for tri-modal classification.
- Experiment with **transformer-based fusion layers** instead of simple concatenation.

