import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # hides noisy TensorFlow logs

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

MODEL_PATH = "best_freshness_model.keras"
IMG_SIZE = 128  # must match the size used in training

st.set_page_config(page_title="FreshHarvest Inspector", page_icon="🍓")


# Load the model once and reuse it. Without this cache, Streamlit would
# reload the model from disk every time the user interacts with the page.
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)



def predict(image):
    image = image.convert("RGB")

    # Match the trained model's input dimensions
    image = image.resize((128, 128))

    arr = np.array(image, dtype="float32")

    # Keep preprocessing consistent with training
    arr = preprocess_input(arr)

    arr = np.expand_dims(arr, axis=0)

    prediction = model.predict(arr, verbose=0)

    p_spoiled = float(prediction[0][0])

    if p_spoiled > 0.5:
        return "SPOILED", p_spoiled

    return "FRESH", 1 - p_spoiled


st.title("🍎 FreshHarvest Freshness Inspector")
st.write("Drag and drop a photo of a fruit or vegetable to check whether it is **fresh** or **spoiled**.")

model = load_model()

uploaded = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing..."):
        label, confidence = predict(image)

    if label == "FRESH":
        st.success(f"Prediction: **{label}**  (confidence: {confidence:.1%})")
    else:
        st.error(f"Prediction: **{label}**  (confidence: {confidence:.1%})")

    st.progress(confidence)