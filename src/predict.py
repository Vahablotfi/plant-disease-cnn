"""
Predict plant disease class for one image using the trained custom CNN.

Run from the project root:

    python src/predict.py path/to/image.jpg
"""

# Code explanation:
# argparse reads the image path from the terminal.
# json loads the saved class names.
# pathlib helps build file paths that work cleanly on macOS, Linux, and Colab.

# Exam note:
# This script is the inference/prediction step. Training has already happened,
# so here we only load the saved model and use it to predict one new image.

import argparse
import json
from pathlib import Path


# Code explanation:
# These values must match the settings used during training in the notebook.
# The CNN was trained with 128x128 RGB images and pixel values scaled to 0-1.

# Exam note:
# Prediction preprocessing must match training preprocessing. If the model was
# trained on 128x128 normalized images, it should also receive 128x128 normalized
# images during prediction. Otherwise, the input format is different from what
# the CNN learned from, and predictions can become unreliable.

IMG_SIZE = (128, 128)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "simple_custom_cnn_plant_disease.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "class_names.json"


def load_class_names(class_names_path):
    # Code explanation:
    # class_names.json stores the class label order used by the model.
    # The prediction output is an index, so we need this list to convert
    # that index into a readable class name.

    # Exam note:
    # The class order must match the order from training. If the order changes,
    # index 0 could point to the wrong disease name.

    with open(class_names_path, "r", encoding="utf-8") as file:
        return json.load(file)


def prepare_image(image_path):
    try:
        import numpy as np
        import tensorflow as tf
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Missing prediction dependency. Install project requirements first with: "
            "pip install -r requirements.txt"
        ) from error

    # Code explanation:
    # load_img loads the image and resizes it to the same shape used during training.
    # RGB means the image has 3 color channels: red, green, and blue.

    # Exam note:
    # Resizing is needed because neural networks expect a fixed input shape.
    # Our CNN expects images with shape 128x128x3.

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE,
        color_mode="rgb",
    )

    # Code explanation:
    # img_to_array converts the image into a NumPy array of numbers.
    # A color image becomes an array with shape: height, width, channels.

    image_array = tf.keras.utils.img_to_array(image)

    # Code explanation:
    # During training, ImageDataGenerator used rescale=1./255.
    # This changed pixel values from 0-255 to 0-1.
    # We do the same here so prediction input matches training input.

    # Exam note:
    # Normalization helps the model work with smaller, more stable numbers.
    # The exact same normalization should be used in training and prediction.

    image_array = image_array / 255.0

    # Code explanation:
    # The model expects a batch of images, even if we only predict one image.
    # np.expand_dims adds a new first dimension.
    # Shape changes from (128, 128, 3) to (1, 128, 128, 3).

    # Exam note:
    # The batch dimension is needed because Keras models process inputs as batches.
    # A batch size of 1 means we are predicting one image.

    image_batch = np.expand_dims(image_array, axis=0)

    return image_batch


def predict_image(image_path):
    try:
        import numpy as np
        import tensorflow as tf
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Missing prediction dependency. Install project requirements first with: "
            "pip install -r requirements.txt"
        ) from error

    # Code explanation:
    # Before loading files, we check that they exist.
    # This gives a clearer error message if the model, class names, or image path is wrong.

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find model file: {MODEL_PATH}")

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Could not find class names file: {CLASS_NAMES_PATH}")

    if not image_path.exists():
        raise FileNotFoundError(f"Could not find image file: {image_path}")

    # Code explanation:
    # load_model loads the trained CNN saved from Google Colab.
    # The model already contains the learned weights from training.

    # Exam note:
    # Loading the model means we reuse the trained CNN instead of training again.

    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_names(CLASS_NAMES_PATH)
    image_batch = prepare_image(image_path)

    # Code explanation:
    # model.predict returns probability scores for all possible classes.
    # For one image and 38 classes, the shape will usually be (1, 38).
    # Example: [[0.01, 0.03, 0.80, ...]]

    # Exam note:
    # Because the output layer uses Softmax, the model returns a probability-like
    # score for each class. The class with the highest score is the prediction.

    predictions = model.predict(image_batch)
    probabilities = predictions[0]

    # Code explanation:
    # np.argmax finds the index of the largest probability.
    # If index 2 has the largest value, the predicted class is class_names[2].

    # Exam note:
    # np.argmax selects the class that the model thinks is most likely.

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]

    # Code explanation:
    # The confidence score is the highest Softmax probability.
    # Multiplying by 100 converts it to a percentage.

    # Exam note:
    # Confidence is not the same as guaranteed correctness. It only shows how
    # strongly the model prefers the predicted class compared with the others.

    confidence = float(probabilities[predicted_index])

    return predicted_class, confidence


def main():
    # Code explanation:
    # This block defines the terminal interface.
    # The user must provide one image path after the script name.

    parser = argparse.ArgumentParser(
        description="Predict plant disease class for one image using the trained CNN."
    )
    parser.add_argument(
        "image_path",
        help="Path to the image file, for example: path/to/image.jpg",
    )

    args = parser.parse_args()
    image_path = Path(args.image_path)

    predicted_class, confidence = predict_image(image_path)

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence score: {confidence:.4f} ({confidence * 100:.2f}%)")


if __name__ == "__main__":
    main()
