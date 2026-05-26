# plant-disease-cnn

Plant Disease Detection using a simple custom Convolutional Neural Network (CNN).

This is a 4th semester Computer Science machine learning exam project at EK Copenhagen.

## Project Goal

The goal is to classify plant leaf images into disease categories using CNN image classification.

The project currently focuses on a beginner-friendly custom CNN, not transfer learning or advanced pretrained models.

## Current Status

The first notebook works in Google Colab and can:

- download the Kaggle dataset using the Kaggle API
- load the plant disease image folders
- check class names and image counts
- preprocess images with `ImageDataGenerator`
- build a simple custom CNN
- train the model
- evaluate validation accuracy and loss
- print an optional classification report
- save the trained model

Current validation accuracy from the first successful run: around 89.89%.

## Dataset

Dataset used:

`vipoooool/new-plant-diseases-dataset`

The dataset is downloaded inside Google Colab using the Kaggle API.

The dataset is **not included in this GitHub repository** because it is large and should be downloaded directly from Kaggle.

## Project Structure

```text
plant-disease-cnn/
├── README.md
├── requirements.txt
├── .gitignore
├── AGENTS.md
├── notebooks/
│   └── 01_simple_custom_cnn_plant_disease.ipynb
├── src/
│   └── .gitkeep
├── app/
│   └── .gitkeep
├── models/
│   └── .gitkeep
└── data/
    └── .gitkeep
```

## Main Implementation

The main implementation is currently in:

`notebooks/01_simple_custom_cnn_plant_disease.ipynb`

This notebook is written with beginner-friendly comments and exam notes explaining the most important machine learning concepts.

## Model Scope

The model uses:

- `Conv2D`
- `MaxPool2D`
- `Flatten`
- `Dense`
- ReLU activation
- Softmax activation
- `ImageDataGenerator`
- categorical crossentropy

The project does **not** currently use transfer learning, ResNet, EfficientNet, MobileNet, YOLO, transformers, or other advanced models.

## Planned Refinements

These will be added later:

- Streamlit app
- accuracy/loss visualizations
- confusion matrix
- `ModelCheckpoint`
- more detailed README sections

## Notes

The `data/` and `models/` folders are kept in the repository with `.gitkeep` files only.

Actual dataset files and trained model files are ignored by Git for now.
