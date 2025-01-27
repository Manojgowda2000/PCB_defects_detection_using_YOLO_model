# from ultralytics import YOLO
# import cv2
# import matplotlib.pyplot as plt

# # Load the pre-trained YOLOv8 model (you can use yolov8n, yolov8s, yolov8m, etc.)
# model = YOLO('newtransferLearning.pt')  # Use the nano model for faster inference

# # Load the image of the PCB
# image_path = 'TESTIMAGE_1005.png'
# image = cv2.imread(image_path)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Perform inference
# results = model(image_path)

# # Display the results
# # The results object contains predictions such as bounding boxes, confidence scores, and class labels
# annotated_image = results[0].plot()  # Annotated image with bounding boxes and labels

# # Show the image with matplotlib
# plt.figure(figsize=(10, 10))
# plt.imshow(annotated_image)
# plt.axis('off')
# plt.show()

# # Print the results
# for result in results[0].boxes.data:
#     x1, y1, x2, y2, conf, cls = result.tolist()
#     print(f"Class: {int(cls)}, Confidence: {conf:.2f}, Box: [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]")












# from ultralytics import YOLO
# import cv2
# import matplotlib.pyplot as plt

# # Load the pre-trained YOLOv8 model (you can use yolov8n, yolov8s, yolov8m, etc.)
# model = YOLO('newtransferLearning.pt')  # Use the nano model for faster inference

# # Load the image of the PCB
# image_path = 'Gateway Board_1004.png'
# image = cv2.imread(image_path)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Perform inference
# results = model(image_path)

# # Create a filtered results object containing only bounding boxes and class labels
# filtered_results = []
# for box in results[0].boxes.data:
#     x1, y1, x2, y2, _, cls = box.tolist()  # Ignore confidence score (_)
#     filtered_results.append({"bbox": [x1, y1, x2, y2], "class": int(cls)})

# # Display the filtered results
# print("Filtered Results:")
# for res in filtered_results:
#     print(f"Class: {res['class']}, Box: {res['bbox']}")

# # Annotate the image with labels only (no confidence scores)
# annotated_image = image_rgb.copy()
# for res in filtered_results:
#     x1, y1, x2, y2 = map(int, res["bbox"])  # Convert coordinates to integers
#     label = model.names[res["class"]]  # Get the class label

#     # Draw the bounding box
#     cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box

#     # Put the label (class name only)
#     cv2.putText(
#         annotated_image,
#         label,
#         (x1, y1 - 10),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.5,  # Font size
#         (255, 0, 0),  # Blue text
#         1  # Thickness
#     )

# # Show the annotated image with matplotlib
# plt.figure(figsize=(10, 10))
# plt.imshow(annotated_image)
# plt.axis('off')
# plt.show()




from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import random

# Load the pre-trained YOLOv8 model
model = YOLO('models/transferLearningRoboModel.pt')  # Use the nano model for faster inference

# Load the image of the PCB
image_path = 'Sparkfun-Dev_1002.png'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform inference
results = model(image_path,conf=0.3)

# Create a filtered results object containing only bounding boxes and class labels
filtered_results = []
for box in results[0].boxes.data:
    x1, y1, x2, y2, _, cls = box.tolist()  # Ignore confidence score (_)
    filtered_results.append({"bbox": [x1, y1, x2, y2], "class": int(cls)})

# Generate unique colors for each class
class_colors = {cls: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for cls in range(len(model.names))}

# Annotate the image with labels and unique background colors
annotated_image = image_rgb.copy()
for res in filtered_results:
    x1, y1, x2, y2 = map(int, res["bbox"])  # Convert coordinates to integers
    label = model.names[res["class"]]  # Get the class label
    color = class_colors[res["class"]]  # Get the color for this class

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
        0.5,  # Font size
        (255, 255, 255),  # White text
        1  # Thickness
    )

# Show the annotated image with matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(annotated_image)
plt.axis('off')
plt.show()

# Display the filtered results
print("Filtered Results:")
for res in filtered_results:
    print(f"Class: {res['class']}, Box: {res['bbox']}")