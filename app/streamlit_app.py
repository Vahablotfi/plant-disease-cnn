"""
Simple Streamlit app for plant disease prediction.

Run from the project root:

    streamlit run app/streamlit_app.py
"""

# Code explanation:
# json loads the class names saved after training.
# pathlib helps create file paths that work from the project root.
# numpy is used for array operations such as adding the batch dimension.
# streamlit creates the simple web app.
# tensorflow loads the trained CNN model and prepares the uploaded image.

# Exam note:
# This app is an inference interface. The CNN has already been trained in the
# notebook, so the app only loads the saved model and uses it for prediction.

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# Code explanation:
# These settings must match the notebook and src/predict.py.
# The model was trained with 128x128 RGB images and pixel values scaled to 0-1.

# Exam note:
# Prediction preprocessing must match training preprocessing. If the CNN learned
# from 128x128 normalized images, the app must send images in that same format.
# Otherwise, the model receives data in a different form than it learned from.

IMG_SIZE = (128, 128)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "simple_custom_cnn_plant_disease.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "class_names.json"


@st.cache_resource
def load_model():
    # Code explanation:
    # Streamlit reruns the script whenever the user interacts with the page.
    # cache_resource keeps the model loaded so it does not reload on every rerun.

    # Exam note:
    # Loading the model means we reuse the trained CNN weights instead of
    # training the model again inside the app.

    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    # Code explanation:
    # class_names.json stores the class label order used during training.
    # The model returns class indexes, and this list converts indexes into names.

    # Exam note:
    # The class order must stay the same as during training. If the order changes,
    # the app could show the wrong disease name for a prediction index.

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def preprocess_uploaded_image(uploaded_image):
    # Code explanation:
    # Streamlit gives the uploaded image as a file-like object.
    # PIL opens it as an image, and convert("RGB") makes sure it has 3 color channels.

    # Exam note:
    # The CNN expects RGB images because it was trained with color leaf images.

    image = Image.open(uploaded_image).convert("RGB")

    # Code explanation:
    # Display image is kept separately so the user sees the original uploaded image.
    # Prediction image is resized to the exact size expected by the CNN.

    display_image = image.copy()
    prediction_image = image.resize(IMG_SIZE)

    # Code explanation:
    # img_to_array converts the image into numbers.
    # Then dividing by 255.0 normalizes pixels from 0-255 to 0-1.

    # Exam note:
    # Normalization must match training. The notebook used ImageDataGenerator
    # with rescale=1./255, so the app uses the same scaling.

    image_array = tf.keras.utils.img_to_array(prediction_image)
    image_array = image_array / 255.0

    # Code explanation:
    # The model expects a batch of images.
    # np.expand_dims changes the shape from (128, 128, 3) to (1, 128, 128, 3).

    # Exam note:
    # The batch dimension is needed even for one image because Keras models
    # expect input in the format: batch size, height, width, channels.

    image_batch = np.expand_dims(image_array, axis=0)

    return display_image, image_batch


def predict(image_batch, model, class_names):
    # Code explanation:
    # model.predict returns one probability score for each class.
    # With one uploaded image and 38 classes, the result shape is usually (1, 38).

    # Exam note:
    # The output layer uses Softmax, so the scores can be read as confidence-like
    # values. The largest value is selected as the predicted class.

    predictions = model.predict(image_batch)
    probabilities = predictions[0]

    # Code explanation:
    # np.argmax finds the index of the highest probability.
    # That index is used to look up the readable disease class name.

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    return predicted_class, confidence


def main():
    st.title("Plant Disease CNN")
    st.write("Upload a plant leaf image and the trained CNN will predict the disease class.")

    st.warning(
        "This model is for educational purposes and is not a professional agricultural diagnosis tool."
    )

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    if not CLASS_NAMES_PATH.exists():
        st.error(f"Class names file not found: {CLASS_NAMES_PATH}")
        st.stop()

    model = load_model()
    class_names = load_class_names()

    # Code explanation:
    # file_uploader lets the user choose an image from their computer.
    # The accepted file types are common image formats.

    # Exam note:
    # The uploaded image is new input data. The app preprocesses it the same way
    # as the training images before sending it to the CNN.

    uploaded_file = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("Please upload a JPG or PNG image to get a prediction.")
        return

    display_image, image_batch = preprocess_uploaded_image(uploaded_file)

    st.image(display_image, caption="Uploaded image", use_container_width=True)

    predicted_class, confidence = predict(image_batch, model, class_names)

    st.subheader("Prediction Result")
    st.write(f"**Predicted disease class:** {predicted_class}")
    st.write(f"**Confidence score:** {confidence:.4f} ({confidence * 100:.2f}%)")


if __name__ == "__main__":
    main()
