import streamlit as st
import torch
import cv2
import time

# Load YOLOv7 model
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

    # Initialize detection state in session state
    if 'person_detected' not in st.session_state:
        st.session_state.person_detected = False
    if 'detection_time' not in st.session_state:
        st.session_state.detection_time = 0

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

        # Initialize flag to check if a person is detected
        person_detected = False

        # Process results
        for det in results.xyxy[0]:
            x_min, y_min, x_max, y_max, confidence, class_id = map(int, det[:6])
            label = model.names[class_id]

            if label == "person":
                person_detected = True  # Person detected

                # Draw bounding box and label for the person (still in BGR format)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f'{label}: {confidence:.2f}', (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert frame back to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Check if a person is detected and update session state accordingly
        if person_detected and not st.session_state.person_detected:
            st.session_state.person_detected = True
            st.session_state.detection_time = current_time  # Record the detection time
            st.warning('Person detected!')  # Trigger warning pop-up

        elif st.session_state.person_detected:
            # Reset detection state after 5 seconds
            if current_time - st.session_state.detection_time > 5:
                st.session_state.person_detected = False

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
