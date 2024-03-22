# Product Image Classifier

This repository contains code for a product image classifier, designed to classify images into different product categories. The classifier is built using convolutional neural networks (CNNs) and utilizes state-of-the-art architectures for efficient and accurate classification.

## Video Link
[Click here](https://drive.google.com/drive/folders/1lAIpGDvmR_vQHA-0keOgKxM1DHE64wtA?usp=sharing) to access the video demonstration of the product image classifier.

## Dataset Overview
The dataset from the slash application.it comprises images categorized into different product classes, including Accessories, Beauty, Fashion, Home, Games, Stationary, and Nutrition. However, the dataset suffers from imbalanced class distribution, with Fashion dominating and other classes having significantly fewer samples.
- **Number of classes:** 7
- **Total number of images:** 768
- **Number of images per class:**
  - Accessories: 83
  - Beauty: 47
  - Fashion: 359
  - Games: 36
  - Home: 112
  - Nutrition: 29
  - Stationary: 102

## Dataset Structure
The dataset is structured as follows:
product-images
│
└── slash-data
    ├── Accessories
    │   ├── Accessories1.jpg
    │   └── ...
    │
    ├── Beauty
    │   ├── Beauty1.jpg
    │   └── ...
    │
    ├── Fashion
    │   ├── Fashion1.jpg
    │   └── ...
    │
    ├── Home
    │   ├── Home1.jpg
    │   └── ...
    │
    ├── Games
    │   ├── Games1.jpg
    │   └── ...
    │
    ├── Stationary
    │   ├── Stationary1.jpg
    │   └── ...
    │
    └── Nutrition
        ├── Nutrition1.jpg
        └── ...


## Preprocessing Images
The following preprocessing steps are applied to the images:
- **Image Transformations:** Images are resized to 224x224 pixels, converted to tensors, and normalized to have values between -1 and 1.
- **Balancing:** Oversampling is implemented to address class imbalance, ensuring all classes have equal representation.
- **Dataset Splitting:** The balanced dataset is divided into training, validation, and testing sets for comprehensive training and evaluation.

## Modified EfficientNet Model Architecture
The `ModifiedEfficientNet` class is a neural network model tailored for multi-class classification tasks. It is based on the EfficientNet architecture with custom modifications.

### ResNetClass Model Components
The `ResNetClasses` model components include:
- **ResNet Backbone:** Consists of convolutional layers, pooling layers, and residual blocks of the ResNet-18 architecture for hierarchical feature extraction.
- **Fully Connected Layer (Classifier):** A single fully connected layer with ReLU activation that transforms the extracted features into logits for each class.

#### Results:
- Average Train Loss: 0.1580
- Validation Accuracy: 91.54%
- Test Accuracy: 93.84%

### ModelCNN Model Components
The `ModelCNN` model components include:
- **Convolutional Layers:** Three convolutional layers with max pooling, ReLU activation, and dropout for feature extraction and spatial dimension reduction.
- **Fully Connected Layers:** Two fully connected layers for classification, with ReLU activation and dropout for regularization.

#### Results:
- Average Train Loss: 0.0015
- Validation Accuracy: 94.78%
- Test Accuracy: 95.03%

This repository provides a detailed insight into the implementation and performance of the product image classifier.
