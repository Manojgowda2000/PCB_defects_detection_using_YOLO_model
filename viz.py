from torchsummary import summary
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('newtransferLearning.pt').model  # Access the underlying PyTorch model

# Display summary for input size (e.g., 640x640 RGB images)
summary(model, input_size=(3, 640, 640))