import streamlit as st
import torch
import cv2
import time

# Load YOLOv7 model
# test on yolov7 , yolov7-w6 , yolov7x
@st.cache_resource()
def load_model():
    # Load the YOLOv7 model from the local weights
    model = torch.hub.load('yolov7', 'custom', 'yolov7.pt', source='local')
    return model

# Function to perform object detection on a frame
def detect_objects(frame, model):
    # Perform inference
    results = model(frame)
    return results

# Streamlit interface
def main():
    st.title('Real-time Object Detection using YOLOv7')

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

        # Convert frame to RGB for detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection
        results = detect_objects(frame_rgb, model)

        # Process results and draw bounding boxes
        for det in results.xyxy[0]:
            x_min, y_min, x_max, y_max = map(int, det[:4])  # Only convert coordinates to int
            confidence = det[4]  # Confidence is a float, don't convert to int
            class_id = int(det[5])  # Convert class ID to int
            label = model.names[class_id]

            # Draw bounding box and label for detected objects
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {confidence:.2f}', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert frame back to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Calculate FPS
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Display FPS and frame
        fps_placeholder.text(f'FPS: {fps:.2f}')
        frame_placeholder.image(frame_rgb, channels='RGB', use_column_width=True)

        run_detection = st.session_state.run_detection

    video_capture.release()

if __name__ == '__main__':
    main()
