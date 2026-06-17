import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
model = load_model("emotion_model.h5")

# Labels
labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

st.set_page_config(page_title="Emotion Detection", page_icon="😃")

st.title("😃 Emotion Detection System (CNN)")
st.write("Upload face image to detect emotion")

uploaded_file = st.file_uploader("Choose Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file).convert("L")  # grayscale
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to array
    img = np.array(image)

    # Resize
    img = cv2.resize(img, (48,48))

    # Normalize
    img = img / 255.0

    # Reshape for CNN (batch, height, width, channel)
    img = img.reshape(1, 48, 48, 1)

    # Prediction
    prediction = model.predict(img)

    emotion = labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Output
    st.success(f"Detected Emotion: {emotion}")
    st.info(f"Confidence: {confidence:.2f}%")

    # Show all probabilities
    st.subheader("Emotion Scores")

    for i, label in enumerate(labels):
        st.write(f"{label}: {prediction[0][i]*100:.2f}%")