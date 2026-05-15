import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="finpro salim deepfake detection",
    page_icon="🧠",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

html, body, [class*="css"]{
    font-family: 'Arial';
    background: #0f172a;
    color: white;
}

.main {
    background: linear-gradient(135deg,#0f172a,#111827);
}

.block-container{
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.title{
    font-size:52px;
    font-weight:800;
    background: linear-gradient(to right,#38bdf8,#8b5cf6);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    margin-bottom:10px;
}

.subtitle{
    font-size:20px;
    color:#cbd5e1;
    margin-bottom:40px;
}

.upload-box{
    background:#1e293b;
    padding:25px;
    border-radius:20px;
    border:1px solid rgba(255,255,255,0.08);
}

.card{
    background: rgba(255,255,255,0.05);
    border-radius:25px;
    padding:25px;
    backdrop-filter: blur(10px);
    border:1px solid rgba(255,255,255,0.08);
}

.result-real{
    background: linear-gradient(135deg,#22c55e,#16a34a);
    padding:35px;
    border-radius:25px;
    text-align:center;
    box-shadow:0 0 30px rgba(34,197,94,0.35);
}

.result-fake{
    background: linear-gradient(135deg,#ef4444,#f97316);
    padding:35px;
    border-radius:25px;
    text-align:center;
    box-shadow:0 0 30px rgba(239,68,68,0.35);
}

.metric-card{
    background: rgba(255,255,255,0.06);
    border-radius:22px;
    padding:25px;
    text-align:center;
    margin-top:20px;
}

.stButton>button{
    width:100%;
    background: linear-gradient(135deg,#3b82f6,#8b5cf6);
    color:white;
    border:none;
    padding:16px;
    border-radius:18px;
    font-size:18px;
    font-weight:bold;
    transition:0.3s;
}

.stButton>button:hover{
    transform:scale(1.03);
    box-shadow:0 0 25px rgba(59,130,246,0.5);
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
st.markdown('<div class="title">🧠 VisionGuard AI</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Advanced Deepfake Detection System using Deep Learning</div>',
    unsafe_allow_html=True
)

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns([1,1])

# =========================
# LEFT SIDE
# =========================
with col1:

    st.markdown('<div class="upload-box">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Upload Image",
        type=["jpg","jpeg","png"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

# =========================
# RIGHT SIDE
# =========================
with col2:

    st.markdown("""
    <div class="card">
        <h1>🔍 AI Detection Panel</h1>
        <p style="font-size:18px;color:#cbd5e1;">
            Analyze whether the uploaded image is authentic or manipulated using a CNN deep learning model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    if uploaded_file:

        analyze = st.button("Analyze Image")

        if analyze:

            with st.spinner("Analyzing Image..."):

                # =========================
                # PREPROCESS
                # =========================
                img = image.resize((224,224))

                img_array = np.array(img)

                img_array = img_array.astype(np.float32) / 255.0

                input_tensor = tf.convert_to_tensor(
                    np.expand_dims(img_array, axis=0)
                )

                # =========================
                # PREDICTION
                # =========================
                output = infer(input_tensor)

                pred = list(output.values())[0].numpy()[0][0]

                # =========================
                # REAL
                # =========================
                if pred > 0.5:

                    confidence = float(pred * 100)

                    st.markdown(
                        f"""
                        <div class="result-real">

                            <h1 style="
                                color:white;
                                font-size:42px;
                                margin-bottom:10px;
                            ">
                                ✅ REAL IMAGE
                            </h1>

                            <p style="
                                color:white;
                                font-size:22px;
                                margin:0;
                            ">
                                Confidence: {confidence:.2f}%
                            </p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    c1, c2 = st.columns(2)

                    with c1:

                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <h3>Authenticity</h3>
                                <h1>{confidence:.1f}%</h1>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with c2:

                        st.markdown(
                            """
                            <div class="metric-card">
                                <h3>Status</h3>
                                <h1>REAL</h1>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.balloons()

                # =========================
                # FAKE
                # =========================
                else:

                    confidence = float((1 - pred) * 100)

                    st.markdown(
                        f"""
                        <div class="result-fake">

                            <h1 style="
                                color:white;
                                font-size:42px;
                                margin-bottom:10px;
                            ">
                                ❌ FAKE IMAGE
                            </h1>

                            <p style="
                                color:white;
                                font-size:22px;
                                margin:0;
                            ">
                                Confidence: {confidence:.2f}%
                            </p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    c1, c2 = st.columns(2)

                    with c1:

                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <h3>Manipulation Risk</h3>
                                <h1>{confidence:.1f}%</h1>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with c2:

                        st.markdown(
                            """
                            <div class="metric-card">
                                <h3>Status</h3>
                                <h1>FAKE</h1>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )