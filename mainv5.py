import streamlit as st
import torch
import cv2
from PIL import Image
import numpy as np
import time

# Load YOLOv5 model
# test other models like yolov5s , yolov5l , yolov5x
@st.cache_data()
@st.cache_resource()
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

# Function to perform object detection on a frame
def detect_objects(frame, model):
    # YOLOv5 expects a NumPy array (BGR image)
    results = model(frame)
    return results.pandas().xyxy[0]

# Streamlit interface
def main():
    st.title('Real-time Object Detection using YOLOv5')

    model = load_model()

    run_detection = st.checkbox('Run Object Detection', key='run_detection')

    video_capture = cv2.VideoCapture(0)

    frame_placeholder = st.empty()
    fps_placeholder = st.empty()

    prev_time = time.time()  # To track time for FPS calculation

    while run_detection:
        ret, frame = video_capture.read()

        if not ret:
            st.write('Error: Cannot capture video. Please check your camera.')
            break

        # YOLOv5 expects the frame in BGR format (OpenCV format is BGR by default)
        # Perform object detection
        detections = detect_objects(frame, model)

        # Draw bounding boxes and labels
        for index, row in detections.iterrows():
            label = row['name']
            confidence = row['confidence']
            x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(frame, f'{label}: {confidence:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Display FPS on the Streamlit app
        fps_placeholder.text(f'FPS: {fps:.2f}')

        # Convert frame to RGB for Streamlit display (Streamlit expects RGB format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels='RGB', use_column_width=True)

        # Update the checkbox state to break the loop if unchecked
        run_detection = st.session_state.run_detection

    video_capture.release()

if __name__ == '__main__':
    main()
