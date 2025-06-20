import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Setup the page
st.set_page_config(
    page_title="AlphaOne - Mango Leaf Classifier",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Green Gradient Theme
st.markdown("""
    <style>
    html, body, [class*="css"] {
        height: auto !important;
        overflow: auto !important;
    }
    body {
        background: linear-gradient(135deg, #d6f5d6, #b2f0c0, #7ddba2, #4caf50, #2e7d32);
        background-size: 400% 400%;
        animation: gradientFlow 15s ease infinite;
        font-family: 'Segoe UI', sans-serif;
    }
    @keyframes gradientFlow {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .main {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2.5rem 2rem;
        border-radius: 24px;
        box-shadow: 0 10px 30px rgba(0, 80, 30, 0.2);
        max-width: 750px;
        margin: 2rem auto;
    }
    h1 {
        color: #1a3d2d;
        font-size: 3em;
        margin-bottom: 0.2em;
        text-align: center;
    }
    h3 {
        color: #326849;
        text-align: center;
        margin-bottom: 1.4em;
    }
    .stFileUploader label {
        display: none;
    }
    .stButton > button {
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        padding: 0.6rem 1.4rem;
        border-radius: 10px;
        transition: 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #25682a;
        transform: scale(1.04);
    }
    .result-box {
        background-color: #ecfdf5;
        border-left: 6px solid #1b5e20;
        padding: 1rem;
        border-radius: 12px;
        margin-top: 1.5rem;
        color: #1b4d3e;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("alphaone.h5", compile=False)

model = load_model()

# Class labels
class_names = [
    'Anthracnose', 'Bacterial Canker', 'Cutting Weevil',
    'Die Back', 'Gall Midge', 'Healthy',
    'Powdery Mildew', 'Sooty Mould'
]

# Main container
with st.container():
    st.markdown('<div class="main">', unsafe_allow_html=True)

    st.markdown("### 🌿")
    st.markdown("<h1>AlphaOne</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Mango Leaf Disease Classifier</h3>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Mango Leaf", use_column_width=True)

        # Preprocess image
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 224, 224, 3)

        # Predict
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = prediction[0][predicted_index] * 100

        # Display results
        st.markdown(f"""
            <div class="result-box">
                <b>Predicted Disease:</b> {predicted_class}<br>
                <b>Confidence:</b> {confidence:.2f}%
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
