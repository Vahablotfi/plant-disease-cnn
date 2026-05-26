# AGENTS.md

## Project Context

This repository is for a 4th semester Computer Science machine learning exam project at EK Copenhagen.

Project name: `plant-disease-cnn`

Main goal: classify plant leaf disease images using a beginner-friendly custom CNN.

## Current Scope

Keep the project within beginner CNN image classification:

- `ImageDataGenerator` / image preprocessing
- `Conv2D`
- `MaxPool2D`
- `Flatten`
- `Dense`
- ReLU
- Softmax
- model training
- model validation/evaluation
- saving the trained model

## Avoid For Now

Do not use these as the main solution:

- transfer learning
- pretrained CNNs such as ResNet, EfficientNet, MobileNet, VGG, Inception
- YOLO
- transformers
- RAG
- LLM fine-tuning
- advanced techniques outside the class scope

## Repository Rules

- Do not commit the Kaggle dataset.
- Do not commit trained model files yet.
- Do not commit `kaggle.json` or other API keys.
- Keep code beginner-friendly with clear comments.
- Keep exam-note comments where useful so the project is easy to explain orally.

## Current Main Implementation

The current implementation is in:

`notebooks/01_simple_custom_cnn_plant_disease.ipynb`

Future refinements may add:

- Streamlit app in `app/`
- training plots
- confusion matrix
- `ModelCheckpoint`
- more detailed documentation
