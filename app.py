import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ✅ This MUST be the FIRST Streamlit command
st.set_page_config(page_title="Mango Leaf Classifier", layout="centered")

# Load the trained model only once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("alphaone.h5", compile=False)

model = load_model()

# Class labels
class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
               'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

# UI content
st.title("🍃 Mango Leaf Disease Classifier (alphaone)")
st.markdown("Upload a mango leaf image and detect the disease using the AI model.")

# Image uploader
uploaded_file = st.file_uploader("📷 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image.resize((224, 224))) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    st.markdown("## 🧪 Prediction Result")
    st.success(f"**Predicted Class:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")
