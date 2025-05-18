# Library imports
import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model  # ✅ Use tensorflow.keras instead of keras

# Load model
model = load_model('plant_disease_model.h5')

# Class names
CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Barly blight', 'Corn-Common_rust')

# App UI
st.title("Plant Disease Detection")
st.markdown("Upload an image of the plant leaf")

# Upload image
plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict Disease')

# Prediction logic
if submit:
    if plant_image is not None:
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        st.image(opencv_image, channels="BGR")
        st.write("Image Shape:", opencv_image.shape)

        opencv_image = cv2.resize(opencv_image, (256, 256))
        opencv_image = opencv_image / 255.0  # ✅ Normalize the image if needed
        opencv_image = np.reshape(opencv_image, (1, 256, 256, 3))

        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        st.success(f"This is a **{result.split('-')[0]}** leaf with **{result.split('-')[1]}**.")
