import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="DeepShield AI",
    page_icon="🛡️",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

body {
    background-color: #0f172a;
}

.main {
    background: linear-gradient(180deg,#0f172a,#111827);
    color: white;
}

.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    margin-top: 10px;
    color: white;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
}

.real {
    background-color: rgba(34,197,94,0.15);
    border: 1px solid #22c55e;
    color: #22c55e;
}

.fake {
    background-color: rgba(239,68,68,0.15);
    border: 1px solid #ef4444;
    color: #ef4444;
}

.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    background-color: #2563eb;
    color: white;
    border: none;
    font-size: 16px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("deepfake_savedmodel")

# =========================
# HEADER
# =========================
st.markdown('<div class="title">🛡️ DeepShield AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Advanced Deepfake Detection System using CNN</div>',
    unsafe_allow_html=True
)

# =========================
# UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload Face Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION
# =========================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.image(image, use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    confidence = float(max(pred, 1-pred) * 100)

    if pred > 0.5:

        st.markdown(
            f"""
            <div class="result-box real">
                ✅ REAL IMAGE<br>
                Confidence: {confidence:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(int(confidence))

    else:

        st.markdown(
            f"""
            <div class="result-box fake">
                ❌ FAKE IMAGE<br>
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