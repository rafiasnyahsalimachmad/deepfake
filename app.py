import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="VisionGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg,#0F172A,#111827,#1E293B);
    color: white;
}

/* HEADER */
.header-box {
    padding: 35px;
    border-radius: 28px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(16px);
    margin-bottom: 25px;
}

.main-title {
    font-size: 62px;
    font-weight: 700;
    background: linear-gradient(to right,#38BDF8,#818CF8,#C084FC);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

.sub-title {
    color: #CBD5E1;
    font-size: 18px;
}

/* CARD */
.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 28px;
    padding: 25px;
    backdrop-filter: blur(14px);
}

/* BUTTON */
.stButton > button {
    width: 100%;
    height: 58px;
    border-radius: 18px;
    border: none;
    background: linear-gradient(to right,#3B82F6,#8B5CF6);
    color: white;
    font-size: 18px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 0 25px rgba(139,92,246,0.5);
}

/* RESULT BOX */
.real-box {
    background: linear-gradient(135deg,#00C853,#64DD17);
    padding: 28px;
    border-radius: 24px;
    text-align: center;
    color: white;
    box-shadow: 0 0 25px rgba(0,255,120,0.35);
    margin-top: 15px;
}

.fake-box {
    background: linear-gradient(135deg,#FF1744,#FF9100);
    padding: 28px;
    border-radius: 24px;
    text-align: center;
    color: white;
    box-shadow: 0 0 25px rgba(255,80,80,0.35);
    margin-top: 15px;
}

.result-title {
    font-size: 34px;
    font-weight: 700;
}

.result-confidence {
    font-size: 22px;
    margin-top: 12px;
}

/* METRIC */
.metric-card {
    background: rgba(255,255,255,0.06);
    padding: 18px;
    border-radius: 20px;
    text-align: center;
    margin-top: 15px;
}

/* FOOTER */
.footer {
    text-align: center;
    color: #94A3B8;
    margin-top: 50px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    model = tf.saved_model.load("deepfake_savedmodel")
    return model

model = load_model()
infer = model.signatures["serving_default"]

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="header-box">
    <div class="main-title">🛡️ VisionGuard AI</div>
    <div class="sub-title">
        Deepfake detection system powered by CNN Deep Learning
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# LAYOUT
# =====================================================
left_col, right_col = st.columns([1.1, 1])

# =====================================================
# LEFT PANEL
# =====================================================
with left_col:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# RIGHT PANEL
# =====================================================
with right_col:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🔍 AI Detection Panel")

    st.write(
        "Analyze whether the uploaded image is authentic or manipulated."
    )

    if uploaded_file is not None:

        if st.button("Analyze Image"):

            # loading animation
            progress = st.progress(0)

            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

            # =====================================================
            # PREPROCESS
            # =====================================================
            img = image.resize((128,128))

            img_array = np.array(img).astype(np.float32)

            img_array = img_array / 255.0

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            input_tensor = tf.convert_to_tensor(img_array)

            # =====================================================
            # PREDICT
            # =====================================================
            output = infer(input_tensor)

            pred = list(output.values())[0].numpy()[0][0]

            # =====================================================
            # REAL IMAGE
            # =====================================================
            if pred > 0.5:

                confidence = float(pred * 100)

                st.markdown(
                    f"""
                    <div class="real-box">

                        <div class="result-title">
                            ✅ REAL IMAGE
                        </div>

                        <div class="result-confidence">
                            Confidence: {confidence:.2f}%
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                c1, c2 = st.columns(2)

                with c1:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h4>Authenticity</h4>
                            <h2>{confidence:.1f}%</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with c2:

                    st.markdown(
                        """
                        <div class="metric-card">
                            <h4>Status</h4>
                            <h2>REAL</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.balloons()

            # =====================================================
            # FAKE IMAGE
            # =====================================================
            else:

                confidence = float((1 - pred) * 100)

                st.markdown(
                    f"""
                    <div class="fake-box">

                        <div class="result-title">
                            ❌ FAKE IMAGE
                        </div>

                        <div class="result-confidence">
                            Confidence: {confidence:.2f}%
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                c1, c2 = st.columns(2)

                with c1:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h4>Manipulation Risk</h4>
                            <h2>{confidence:.1f}%</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with c2:

                    st.markdown(
                        """
                        <div class="metric-card">
                            <h4>Status</h4>
                            <h2>FAKE</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    else:

        st.info("Upload an image to begin analysis.")

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
    VisionGuard AI • TensorFlow • Streamlit
</div>
""", unsafe_allow_html=True)