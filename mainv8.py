import streamlit as st
import cv2
from PIL import Image
import numpy as np
import time
from ultralytics import YOLO

# Load YOLOv8 model
@st.cache_resource()
def load_model():
    model = YOLO('yolov8x.pt')  # Load YOLOv8 model
    return model

# Function to perform object detection on a frame
def detect_objects(frame, model):
    results = model(frame)
    return results

# Streamlit interface
def main():
    st.title('Real-time Object Detection using YOLOv8')

    model = load_model()

    run_detection = st.checkbox('Run Object Detection', key='run_detection')

    video_capture = cv2.VideoCapture(0)

    frame_placeholder = st.empty()
    fps_placeholder = st.empty()

    prev_time = 0

    while run_detection:
        ret, frame = video_capture.read()

        if not ret:
            st.write('Error: Cannot capture video. Please check your camera.')
            break

        current_time = time.time()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)

        
        results = detect_objects(pil_img, model)

        # Draw bounding boxes and labels
        for result in results:
            boxes = result.boxes  # YOLOv8 stores boxes in results.boxes
            for box in boxes:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])  # Extract box coordinates
                confidence = box.conf[0]  # Confidence score
                label = model.names[int(box.cls[0])]  # Class label

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(frame, f'{label}: {confidence:.2f}', (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Calculate FPS
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Display FPS on the Streamlit app
        fps_placeholder.text(f'FPS: {fps:.2f}')

        # Display frame
        frame_placeholder.image(frame, channels='RGB', use_column_width=True)

        # Update the checkbox state to break the loop if unchecked
        run_detection = st.session_state.run_detection

    video_capture.release()

if __name__ == '__main__':
    main()
