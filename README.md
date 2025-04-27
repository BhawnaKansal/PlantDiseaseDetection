# Plant Disease Classification using Neural Networks

## Overview

This project provides an end-to-end solution for classifying plant leaf diseases using deep learning. Leveraging the PlantVillage dataset, a Convolutional Neural Network (CNN) is trained to recognize and predict various plant diseases from images. The workflow includes data acquisition, preprocessing, model training, evaluation, and prediction utilities.

---

## Table of Contents

- [Project Motivation](#project-motivation)
- [Why Neural Networks?](#why-neural-networks)
- [Dataset](#dataset)
- [Installation & Setup](#installation--setup)
- [Code Structure](#code-structure)
- [How to Run](#how-to-run)
- [Model Architecture](#model-architecture)
- [Results & Evaluation](#results--evaluation)
- [Usage Example](#usage-example)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

---

## Project Motivation

Early and accurate detection of plant diseases is crucial for global food security and sustainable agriculture. Manual inspection is time-consuming and prone to human error. This project aims to automate disease detection using image classification, making it accessible and scalable for farmers and researchers.

---

## Why Neural Networks?

Neural networks, especially Convolutional Neural Networks (CNNs), are state-of-the-art for image classification tasks. They excel at automatically learning hierarchical patterns and features from raw pixel data, which is essential for distinguishing subtle differences between healthy and diseased leaves. Unlike traditional machine learning, which often requires manual feature extraction, neural networks can learn complex representations directly from images, improving accuracy and scalability as more data is provided[3][5][6].

---

## Dataset

- **Source:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Contents:** Thousands of labeled images of healthy and diseased plant leaves across multiple crop species.
- **Structure:** Images are organized by crop and disease type. The project uses the color images for training and evaluation.

---

## Installation & Setup

1. **Clone the repository and navigate to the project directory.**
2. **Install dependencies:**
    ```
    pip install tensorflow numpy matplotlib pillow kaggle
    ```
3. **Kaggle API Setup:**
    - Download `kaggle.json` from your Kaggle account.
    - Place it in the working directory.

4. **Download the dataset:**
    - The notebook will automatically download and extract the PlantVillage dataset using Kaggle API.

---

## Code Structure

- `PlantDisease.ipynb` - Main notebook containing all code and explanations.
- `plantvillage-dataset.zip` - Downloaded dataset archive.
- `plantvillage dataset/` - Extracted dataset directory.
- `class_indices.json` - Mapping of class indices to class names.
- `plant_disease_prediction_model.h5` - Saved trained model.
- `kaggle.json` - Kaggle API credentials (not shared).

---

## How to Run

1. **Open `PlantDisease.ipynb` in Jupyter or Colab.**
2. **Run all cells sequentially:**
    - Installs and configures the Kaggle API.
    - Downloads and extracts the dataset.
    - Explores and visualizes images.
    - Preprocesses data and creates training/validation splits.
    - Builds and trains the CNN model.
    - Plots training and validation accuracy/loss.
    - Saves the trained model and class indices.
3. **Use the provided functions to predict disease class for new images.**

---

## Model Architecture

- **Input:** 224x224 RGB images.
- **Layers:**
    - Conv2D (32 filters, 3x3, ReLU) + MaxPooling2D
    - Conv2D (64 filters, 3x3, ReLU) + MaxPooling2D
    - Flatten
    - Dense (256 units, ReLU)
    - Dense (number of classes, Softmax)
- **Loss:** Categorical Crossentropy
- **Optimizer:** Adam
- **Metrics:** Accuracy

---

## Results & Evaluation

- The model achieves strong validation accuracy (see notebook for exact results).
- Training and validation accuracy/loss are plotted for visual inspection.
- The model can generalize well to unseen images, demonstrating the effectiveness of CNNs for this task.

---

## Future Work

- Increase training epochs and use data augmentation for improved accuracy.
- Experiment with deeper architectures or transfer learning (e.g., ResNet, EfficientNet).
- Deploy as a web/mobile app for real-world usage.
- Expand to other crops and disease datasets.

---

## Acknowledgements

- [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- TensorFlow and Keras teams
- Kaggle for dataset hosting

---

## Frequently Asked Questions

**Q: Why use a neural network for this problem?**  
A: Neural networks, particularly CNNs, are the best tools for image classification. They can automatically learn complex features from raw images, outperforming traditional ML approaches that require manual feature engineering. This capability is essential for plant disease detection, where subtle visual cues distinguish diseases[3][5][6].

**Q: Can this model be used in the field?**  
A: Yes. With further development and deployment (e.g., as a mobile app), farmers and agronomists can use this tool for real-time disease diagnosis.

---

**Author:**  
*Bhawna Kansal*


