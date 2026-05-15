import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# ====================================
# PAGE CONFIG
# ====================================
st.set_page_config(
    page_title="Deepfake Vision AI",
    page_icon="🎭",
    layout="centered"
)

# ====================================
# CUSTOM CSS
# ====================================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #141E30, #243B55);
    color: white;
}

.title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    background: linear-gradient(to right, #00F5A0, #00D9F5, #A855F7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #D1D5DB;
    font-size: 18px;
    margin-bottom: 30px;
}

.upload-box {
    padding: 20px;
    border-radius: 20px;
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.15);
}

.real-box {
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(135deg, #00C853, #64DD17);
    color: white;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    animation: fadeIn 0.8s ease;
}

.fake-box {
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(135deg, #FF1744, #FF9100);
    color: white;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    animation: fadeIn 0.8s ease;
}

.metric-box {
    padding: 15px;
    border-radius: 15px;
    background: rgba(255,255,255,0.08);
    text-align: center;
    margin-top: 15px;
}

.footer {
    text-align: center;
    color: #B0B0B0;
    margin-top: 50px;
    font-size: 14px;
}

@keyframes fadeIn {
    from {opacity:0; transform: translateY(10px);}
    to {opacity:1; transform: translateY(0);}
}

</style>
""", unsafe_allow_html=True)

# ====================================
# LOAD MODEL
# ====================================
@st.cache_resource
def load_model():
    model = tf.saved_model.load("deepfake_savedmodel")
    return model

model = load_model()
infer = model.signatures["serving_default"]

# ====================================
# HEADER
# ====================================
st.markdown('<div class="title">🎭 Deepfake Vision AI</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Advanced AI system for detecting manipulated images</div>',
    unsafe_allow_html=True
)

# ====================================
# UPLOADER
# ====================================
st.markdown('<div class="upload-box">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

st.markdown('</div>', unsafe_allow_html=True)

# ====================================
# PROCESS IMAGE
# ====================================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🚀 Analyze Image"):

        with st.spinner("Analyzing image with AI..."):

            time.sleep(1.5)

            # preprocessing
            img = image.resize((224, 224))

            img_array = np.array(img) / 255.0

            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

            input_tensor = tf.convert_to_tensor(img_array)

            # prediction
            output = infer(input_tensor)

            pred = list(output.values())[0].numpy()[0][0]

            st.markdown("## 📊 Analysis Result")

            # ====================================
            # REAL
            # ====================================
            if pred > 0.5:

                confidence = float(pred * 100)

                st.markdown(
                    f"""
                    <div class="real-box">
                        ✅ REAL IMAGE
                        <br><br>
                        Confidence: {confidence:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(int(confidence))

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(
                        f"""
                        <div class="metric-box">
                        <h3>Authenticity</h3>
                        <h2>{confidence:.1f}%</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col2:
                    st.markdown(
                        """
                        <div class="metric-box">
                        <h3>Status</h3>
                        <h2>REAL</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.balloons()

            # ====================================
            # FAKE
            # ====================================
            else:

                confidence = float((1 - pred) * 100)

                st.markdown(
                    f"""
                    <div class="fake-box">
                        ❌ FAKE IMAGE
                        <br><br>
                        Confidence: {confidence:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(int(confidence))

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(
                        f"""
                        <div class="metric-box">
                        <h3>Manipulation Risk</h3>
                        <h2>{confidence:.1f}%</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col2:
                    st.markdown(
                        """
                        <div class="metric-box">
                        <h3>Status</h3>
                        <h2>FAKE</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# ====================================
# FOOTER
# ====================================
st.markdown(
    '<div class="footer">Built using TensorFlow • Streamlit • Deep Learning</div>',
    unsafe_allow_html=True
)