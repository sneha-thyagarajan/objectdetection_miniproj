import streamlit as st
import torch
import cv2
from PIL import Image
import numpy as np
import urllib.request

# Load YOLOv5 model
@st.cache_data()
@st.cache_resource()
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    return model

# Function to perform object detection on a frame
def detect_objects(frame, model):
    results = model(frame)
    return results.pandas().xyxy[0]

# Function to get frame from ESP32-CAM
def get_frame_from_esp32(url):
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    frame = cv2.imdecode(imgnp, -1)
    return frame

# Streamlit interface
def main():
    st.title('Real-time Object Detection using YOLOv5')

    model = load_model()

    run_detection = st.checkbox('Run Object Detection', key='run_detection')
    url = 'http://192.168.103.191/cam-hi.jpg'  

    frame_placeholder = st.empty()

    while run_detection:
        frame = get_frame_from_esp32(url)

        if frame is None:
            st.write('Error: Cannot capture video. Please check your camera.')
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)
        
        detections = detect_objects(pil_img, model)

        for index, row in detections.iterrows():
            label = row['name']
            confidence = row['confidence']
            x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(frame, f'{label}: {confidence:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        frame_placeholder.image(frame, channels='RGB', use_column_width=True)

        # Update the checkbox state to break the loop if unchecked
        run_detection = st.session_state.run_detection

if __name__ == '__main__':
    main()


