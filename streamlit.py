import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import random

# Load the YOLO model
model = YOLO('newtransferLearning.pt')

# Function to process and annotate the image
# Function to process and annotate the image
def process_image(image):
    # Convert PIL image to OpenCV format
    image = np.array(image)

    # If the image has an alpha channel (4th channel), remove it
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Perform inference
    results = model(image, conf=0.2)

    # Filter results
    filtered_results = []
    for box in results[0].boxes.data:
        x1, y1, x2, y2, _, cls = box.tolist()
        filtered_results.append({"bbox": [x1, y1, x2, y2], "class": int(cls)})

    # Generate unique colors for each class
    class_colors = {cls: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for cls in range(len(model.names))}

    # Annotate the image
    annotated_image = image.copy()
    for res in filtered_results:
        x1, y1, x2, y2 = map(int, res["bbox"])
        label = model.names[res["class"]]
        color = class_colors[res["class"]]

        # Draw the bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

        # Add a filled rectangle for the label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        label_x1 = x1
        label_y1 = y1 - label_size[1] - 5
        label_x2 = x1 + label_size[0] + 5
        label_y2 = y1
        cv2.rectangle(annotated_image, (label_x1, label_y1), (label_x2, label_y2), color, -1)

        # Put the label text
        cv2.putText(
            annotated_image,
            label,
            (x1 + 2, y1 - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    return annotated_image, filtered_results


# Streamlit UI
st.title("PCB Defect Detection with YOLOv8")
st.write("Upload an image to detect PCB defects using the YOLOv8 model.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)

    # Process the image
    annotated_image, results = process_image(image)

    # Display the uploaded image
    st.subheader("Uploaded Image")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Display the annotated image
    st.subheader("Annotated Image")
    st.image(annotated_image, caption="Annotated Image", use_column_width=True, channels="RGB")

    # Display results
    # st.subheader("Detected Defects")
    # for res in results:
    #     st.write(f"Class: {model.names[res['class']]}, Box: {res['bbox']}")
