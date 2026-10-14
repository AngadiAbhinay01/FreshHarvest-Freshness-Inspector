
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input


# --------------------------------------------------
# 1. App configuration
# --------------------------------------------------
st.set_page_config(
    page_title="FreshHarvest Freshness Inspector",
    page_icon="🍓",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_freshness_model.keras"
IMG_SIZE = 128


# --------------------------------------------------
# 2. Load trained model
# --------------------------------------------------
@st.cache_resource
def load_model(model_path):
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Ensure best_freshness_model.keras is committed "
            "to your GitHub repository."
        )

    return tf.keras.models.load_model(
        str(model_path),
        compile=False
    )


# --------------------------------------------------
# 3. Image prediction
# --------------------------------------------------
def predict(image, model):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert image to NumPy array
    image_array = np.asarray(image, dtype=np.float32)

    # Apply the same preprocessing used during training
    image_array = preprocess_input(image_array)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Get model prediction
    prediction = model.predict(image_array, verbose=0)

    # This assumes a single sigmoid output:
    # 0 = fresh and 1 = spoiled
    if prediction.size != 1:
        raise ValueError(
            "Expected one sigmoid output per image. "
            f"Received output shape: {prediction.shape}"
        )

    spoiled_probability = float(prediction.reshape(-1)[0])

    if not 0.0 <= spoiled_probability <= 1.0:
        raise ValueError(
            "The model output is outside [0, 1]. "
            "Check whether the model uses a sigmoid output."
        )

    if spoiled_probability >= 0.5:
        label = "SPOILED"
        confidence = spoiled_probability
    else:
        label = "FRESH"
        confidence = 1.0 - spoiled_probability

    return label, confidence


# --------------------------------------------------
# 4. App interface
# --------------------------------------------------
st.title("🍎 FreshHarvest Freshness Inspector")

st.write(
    "Upload a fruit or vegetable image to predict "
    "whether it is fresh or spoiled."
)

st.caption(
    "The prediction depends on the trained model and "
    "the image preprocessing used during training."
)


# --------------------------------------------------
# 5. Load model with diagnostic errors
# --------------------------------------------------
try:
    model = load_model(MODEL_PATH)

except Exception as error:
    st.error("Unable to load the trained CNN model.")

    st.write("**Model path:**", str(MODEL_PATH))
    st.write("**File exists:**", MODEL_PATH.is_file())
    st.write("**TensorFlow version:**", tf.__version__)

    with st.expander("View detailed error"):
        st.exception(error)

    st.stop()


# --------------------------------------------------
# 6. Upload image
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# 7. Run prediction
# --------------------------------------------------
if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )

        if st.button("Inspect freshness", type="primary"):
            with st.spinner("Analyzing image..."):
                label, confidence = predict(image, model)

            st.subheader("Inspection Result")

            if label == "FRESH":
                st.success(f"Prediction: {label}")
            else:
                st.error(f"Prediction: {label}")

            st.metric(
                label="Model confidence",
                value=f"{confidence:.1%}"
            )

            st.progress(float(confidence))

            st.info(
                "This is a model prediction, not a guarantee "
                "of food safety."
            )

    except Exception as error:
        st.error("Unable to process this image.")

        with st.expander("View detailed error"):
            st.exception(error)
