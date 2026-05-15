import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🕵️",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    color: #A0A0A0;
    margin-bottom: 30px;
}

.result-real {
    padding: 15px;
    border-radius: 12px;
    background-color: rgba(0,255,100,0.15);
    color: #00FF88;
    font-size: 24px;
    font-weight: bold;
    text-align: center;
}

.result-fake {
    padding: 15px;
    border-radius: 12px;
    background-color: rgba(255,0,80,0.15);
    color: #FF4B6E;
    font-size: 24px;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = tf.saved_model.load("deepfake_savedmodel")
infer = model.signatures["serving_default"]

# =========================
# HEADER
# =========================
st.markdown('<p class="title">🕵️ Deepfake Detector</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">Upload an image and let AI detect whether it is REAL or FAKE</p>',
    unsafe_allow_html=True
)

# =========================
# FILE UPLOADER
# =========================
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION
# =========================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # preprocess
    img = image.resize((224, 224))

    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # predict
    input_tensor = tf.convert_to_tensor(img_array)

    output = infer(input_tensor)

    pred = list(output.values())[0].numpy()[0][0]

    st.write("### Prediction Result")

    # =========================
    # RESULT
    # =========================
    if pred > 0.5:

        confidence = pred * 100

        st.markdown(
            f"""
            <div class="result-real">
                ✅ REAL IMAGE <br>
                Confidence: {confidence:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(int(confidence))

    else:

        confidence = (1 - pred) * 100

        st.markdown(
            f"""
            <div class="result-fake">
                ❌ FAKE IMAGE <br>
                Confidence: {confidence:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(int(confidence))

# =========================
# FOOTER
# =========================
st.markdown("---")

st.caption("Built with TensorFlow & Streamlit")