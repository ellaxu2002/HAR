# Real-Time Sports Highlight Extraction Using Deep Learning

## Overview

This project aims to automate the detection of **sports activities** from static video frames using deep learning techniques. It addresses the challenge of quickly identifying key moments from long surveillance or sports footage, enabling efficient content filtering, highlight generation, and real-time alerting.

The models were trained and evaluated using the **Human Action Recognition (HAR) dataset** from Kaggle, consisting of 12,600 labeled RGB images across 15 action categories.

---

## Problem Statement

In many real-world applications, identifying sports-related actions from still frames is crucial but often time-consuming. This project compares two deep learning architectures to classify each frame as either **“sports”** or **“non-sports”**:

- CNN + Transformer Hybrid
- Hierarchical CNN (CNN_Blk)

The goal is to identify which architecture better balances **recall**, **precision**, and **generalization** for automated highlight extraction systems.

---

## Dataset

- **Source**: Kaggle Human Action Recognition Dataset
- **Size**: 12,600 RGB images (224x224)
  - 9,240 "non-sports" frames
  - 3,360 "sports" frames
- **Preprocessing**:
  - Image resizing to (224, 224)
  - Normalization to [0, 1]
  - Augmentation (zoom, horizontal flip) for training
  - Rescaling for validation/testing
- **Split**: 80% training / 20% validation/testing

---

## Model Architectures

### 1. CNN + Transformer (Benchmark)

A hybrid model that combines **local feature extraction** (via CNN) with **global spatial context modeling** (via Transformer):

- **CNN Backbone**:
  - 3 Conv2D layers with filters [64, 128, 256]
  - Each followed by BatchNorm, ReLU, MaxPooling
- **Transformer Encoder**:
  - Multi-Head Attention
  - Residual connections
  - Layer Normalization
  - Position-wise Feed-Forward MLP
- **Output**:
  - Flattened + Dropout
  - Final sigmoid layer for binary classification

Regularization:
- Dropout: 0.5  
- L2 weight decay: 1e-4  
- Early stopping applied

**Performance**:
- Training Accuracy: ~79.7%
- Validation Accuracy: ~55.3%
- Better **recall**, suitable for high-coverage applications

---

### 2. CNN_Blk (Hierarchical CNN)

A custom CNN designed to extract multi-level spatial features using **concatenation** of shallow and deep layers:

- **Architecture**:
  - Multiple CNN blocks with concatenated feature maps
  - MaxPooling applied to reduce size and focus on salient patterns
- **Training**:
  - Adam optimizer
  - Binary Cross-Entropy with label smoothing (ε = 0.05)
  - Dropout: 0.3 in conv, 0.5 in dense
  - EarlyStopping & ReduceLROnPlateau enabled

**Performance**:
- Training Accuracy: ~88.1%
- Validation Accuracy: ~73.9%
- High **precision**, suitable as second-stage filter

---

## Model Comparison

| Metric     | CNN + Transformer | CNN_Blk         |
|------------|-------------------|-----------------|
| Precision  | Moderate          | **High**        |
| Recall     | **High**          | Low             |
| F1-Score   | Balanced           | Lower due to recall |
| Overfitting Risk | Moderate     | **High**        |

**Recommendation**:
- Use **CNN + Transformer** for primary detection
- Optionally stack **CNN_Blk** as a high-precision filter

---

## Business Impact

- Speeds up sports highlight extraction
- Reduces manual tagging effort
- Improves viewer engagement via faster clip turnaround
- Provides scalable solution for real-time systems
