
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
def predict(image):
    image = image.convert("RGB")
    image = image.resize((128, 128))

    # Pass raw pixel values to the model.
    # The model's Rescaling layer handles scaling.
    arr = np.array(image, dtype="float32")
    arr = np.expand_dims(arr, axis=0)

    prediction = model.predict(arr, verbose=0)
    p_spoiled = float(prediction[0][0])

    if p_spoiled > 0.5:
        return "SPOILED", p_spoiled

    return "FRESH", 1 - p_spoiled

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
