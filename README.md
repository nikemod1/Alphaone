# 🍃 AlphaOne: Mango Leaf Disease Classifier

> A deep learning-powered app that detects diseases in mango leaves using image classification. Built with **TensorFlow** and **Streamlit**.

---

## 🔗 Live App

👉 [classifiermango.streamlit.app](https://classifiermango.streamlit.app)  
Just upload a mango leaf image and get the disease prediction instantly.

---

## 🧠 Features

- ✅ Trained on 8 real mango leaf disease categories
- ✅ Clean UI built with Streamlit
- ✅ Works both offline and online
- ✅ Deploy-ready with minimal setup

---

## 📁 Dataset

Public mango leaf dataset used for training/testing:
[📂 Google Drive Dataset](https://drive.google.com/drive/folders/1-m-_-Z2HlDBJrDbi8QWp48st0U9k5VMp?usp=sharing)

**Classes included:**
- Anthracnose  
- Bacterial Canker  
- Cutting Weevil  
- Die Back  
- Gall Midge  
- Healthy  
- Powdery Mildew  
- Sooty Mould

---

## 🚀 Getting Started (Local Setup)

### 1. Clone this repo

```bash
git clone https://github.com/nikemod1/Alphaone.git
cd Alphaone

python -m venv alphaone-env
alphaone-env\Scripts\activate    # On Windows

pip install -r requirements.txt

streamlit run app.py

Then visit http://localhost:8501 in your browser.

🧪 Model Info
Format: .h5 (Keras)
Input: 224x224 RGB image
Output: 8-class softmax
Framework: TensorFlow 2.17

🙋‍♂️ Author
Made with ❤️ by nikemod1
