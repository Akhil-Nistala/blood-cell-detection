
import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from collections import Counter

# ===========================
# Load Model
# ===========================
model = YOLO("best.pt")

# ===========================
# Helper Function
# ===========================
def detect(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = model.predict(source=img_rgb, imgsz=640, conf=0.25)

    annotated = result[0].plot()
    detections = result[0].boxes.data

    class_names = [model.names[int(cls)] for cls in detections[:, 5]]
    count = Counter(class_names)

    detection_str = ", ".join([f"{k}: {v}" for k, v in count.items()])
    annotated = annotated[:, :, ::-1]

    return annotated, detection_str


# ===========================
# Streamlit UI
# ===========================
st.title("🩸 Blood Cell Detection & Counting (YOLOv10)")
st.write("Upload an image or video, or use webcam for real-time detection.")

conf_threshold = st.slider("Confidence Threshold:", 0.0, 1.0, 0.25, 0.05)

option = st.radio("Choose Input Source:", ["Upload Image", "Upload Video", "Webcam"])


# ===========================
# IMAGE UPLOAD
# ===========================
if option == "Upload Image":
    uploaded_image = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = np.array(cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR))
        annotated, count_str = detect(image)

        st.image(annotated, caption="Annotated Output", use_column_width=True)
        st.success("Detection Counts: " + count_str)


# ===========================
# VIDEO UPLOAD
# ===========================
elif option == "Upload Video":
    uploaded_video = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated, count_str = detect(frame)
            stframe.image(annotated, channels="BGR")

        cap.release()


# ===========================
# WEBCAM MODE
# ===========================
elif option == "Webcam":
    st.write("Turn on webcam and start detection.")

    cam = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        annotated, count_str = detect(frame)

        stframe.image(annotated, channels="BGR")

