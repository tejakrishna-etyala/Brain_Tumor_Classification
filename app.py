import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from keras.preprocessing import image
import numpy as np
import os
import gdown

# Define model path and Google Drive file ID
model_path = "model/my_model.keras"
gdrive_id = "12Z04ZNcaaAalcitn1PxzIHe_UvftqK8T"
model_url = f"https://drive.google.com/uc?id={gdrive_id}"

# Download the model if not exists
if not os.path.exists(model_path):
    st.write("📥 Downloading model from Google Drive...")
    os.makedirs("model", exist_ok=True)
    gdown.download(model_url, model_path, quiet=False)
    st.write("✅ Model downloaded!")

# Load the model
model = tf.keras.models.load_model(model_path)

# Class labels
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Streamlit UI
st.title("🧠 Brain Tumor Classification")

# File uploader
uploaded_file = st.file_uploader("📸 Upload a brain MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=250)

    # Preprocess the image
    img = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 224, 224, 3)
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display result
    st.write(f"### 🎯 Prediction: `{predicted_class}`")
    st.write(f"### 📊 Confidence: `{confidence:.2f}`")
