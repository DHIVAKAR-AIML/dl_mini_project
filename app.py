import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# ------------------------------
# Load Trained Model
# ------------------------------
model = load_model("emotion_model.keras")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ------------------------------
# Streamlit App Layout
# ------------------------------
st.title("😄 Live Face Emotion Detection")
st.write("Webcam stream detecting emotions in real-time")

# Checkbox to start webcam
run = st.checkbox('Start Webcam')

FRAME_WINDOW = st.image([])

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Could not access webcam.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=(0, -1))

        # Predict emotion
        prediction = model.predict(roi_gray, verbose=0)
        emotion = emotion_labels[np.argmax(prediction)]

        # Draw rectangle + label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Show frame in Streamlit
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Release capture when done
cap.release()
