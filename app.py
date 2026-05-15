import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# LOAD MODEL
# =========================
model = tf.saved_model.load("./deepfake_savedmodel")

infer = model.signatures["serve"]

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🧠",
    layout="centered"
)

# =========================
# TITLE
# =========================
st.title("🧠 Deepfake Detection")
st.write("Upload image untuk mendeteksi REAL atau FAKE")

# =========================
# UPLOAD IMAGE
# =========================
uploaded = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION
# =========================
if uploaded is not None:

    # buka image
    image = Image.open(uploaded).convert("RGB")

    # tampilkan image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # preprocess
    img = image.resize((128, 128))

    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    x = tf.convert_to_tensor(x, dtype=tf.float32)

    # predict
    pred = infer(x)

    pred = list(pred.values())[0].numpy()[0][0]

    # =========================
    # LABEL FIXED
    # =========================
    if pred > 0.5:

        confidence = pred * 100

        st.success(
            f"✅ REAL IMAGE ({confidence:.2f}%)"
        )

        st.progress(int(confidence))

    else:

        confidence = (1 - pred) * 100

        st.error(
            f"❌ FAKE IMAGE ({confidence:.2f}%)"
        )

        st.progress(int(confidence))