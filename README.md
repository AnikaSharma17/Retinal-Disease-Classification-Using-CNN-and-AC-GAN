# Retinal-Disease-Classification-Using-CNN-and-AC-GAN
This project uses CNN and AC-GAN to classify retinal diseases like Hypertension, Heart Disease, Cholesterol, and Diabetes from fundus images. AC-GAN helps balance classes by generating synthetic images, improving accuracy and aiding early, automated diagnosis.

## Project Overview
This project focuses on **automated retinal disease classification** using **Convolutional Neural Networks (CNN)** combined with **Auxiliary Classifier Generative Adversarial Networks (AC-GAN)** to enhance model performance by addressing class imbalance.

Retinal diseases like **Hypertension, Heart Disease, Cholesterol disorders, and Diabetes Mellitus** often present subtle changes in retinal fundus images. Manual diagnosis is time-consuming and requires expert ophthalmologists, limiting access in remote areas. This project aims to provide a **fast, accurate, and scalable deep learning solution** for disease detection.

---

## Key Features
- ✅ Multi-class, ternary, and binary classification of retinal diseases.
- ✅ Custom CNN architecture for feature extraction and classification.
- ✅ **AC-GAN** for synthetic image generation to balance class distributions.
- ✅ Data augmentation techniques to improve model generalization.
- ✅ Performance evaluation using Accuracy, Precision, Recall, and AUC.
- ✅ Grad-CAM-based interpretability for model transparency.

---

## Dataset
- Labeled retinal fundus images for five classes:
  - Hypertension
  - Heart Disease
  - Cholesterol
  - Diabetes Mellitus (DM)
  - Control (Healthy)

- **Preprocessing:**
  - Resized images to `128×128` pixels.
  - Applied Histogram Equalization to enhance image contrast.

---

## Methodology
1. **Dataset Preparation**
2. **Image Enhancement** using Histogram Equalization
3. **Data Augmentation**: Rotation, Zoom, Shear, Width/Height Shift, Flip
4. **Model Development**: CNN with Conv2D, BatchNorm, MaxPooling, Dense, Dropout
5. **AC-GAN Integration**: To generate class-specific synthetic images for data balancing
6. **Training and Evaluation**: Accuracy, Precision, Recall, AUC across multi-class, ternary, and binary setups.

---

## Model Architecture
- **CNN Layers:**
  - Convolutional Layers
  - MaxPooling Layers
  - Batch Normalization
  - Fully Connected Layers
  - Softmax Output Layer

- **GAN Type:** Auxiliary Classifier GAN (AC-GAN)

---

## Performance Highlights
- **Binary Classification:** Best performance with AUC ~98%
- **Ternary Classification:** Good generalization with AUC ~95%
- **Five-Class Classification:** Lower performance due to class overlap and data imbalance

---

## Project Structure
```plaintext
├── AC-GAN.py
├── AC-GAN_1.py
├── C2_Final/
├── Final/
├── dataset/
│   ├── train/
│   ├── test/
│   └── val/
├── data_preparation.py
└── README.md

