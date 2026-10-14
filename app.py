import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # hides noisy TensorFlow logs

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models

WEIGHTS_PATH = "cnn_weights.npz"
IMG_SIZE = 128   # the size the CNN was trained on

st.set_page_config(page_title="FreshHarvest Inspector", page_icon="🍓")


@st.cache_resource
def load_model():
    # Same architecture as training, minus the augmentation layers
    # (they do nothing at prediction time).
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),   # probability of "spoiled"
    ])

    data = np.load(WEIGHTS_PATH)
    weights = [data[f"arr_{i}"] for i in range(len(data.files))]

    expected = len(model.get_weights())
    if len(weights) != expected:
        raise ValueError(
            f"Weights file has {len(weights)} arrays but the model needs {expected}. "
            "The architecture in app.py must match the notebook model."
        )
    model.set_weights(weights)
    return model


def predict(image):
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))   # handles PNG alpha and grayscale
    arr = np.expand_dims(np.array(image, dtype="float32"), axis=0)   # raw 0-255; the model rescales itself
    p_spoiled = float(model.predict(arr, verbose=0)[0][0])
    if p_spoiled > 0.5:
        return "SPOILED", p_spoiled
    return "FRESH", 1 - p_spoiled


st.title("🍎 FreshHarvest Freshness Inspector")
st.write("Drag and drop a photo of a fruit or vegetable to check whether it is **fresh** or **spoiled**.")

try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the model weights.")
    st.exception(e)
    st.stop()

uploaded = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing..."):
        label, confidence = predict(image)

    if label == "FRESH":
        st.success(f"Prediction: **{label}** (confidence: {confidence:.1%})")
    else:
        st.error(f"Prediction: **{label}** (confidence: {confidence:.1%})")

    st.progress(confidence)
