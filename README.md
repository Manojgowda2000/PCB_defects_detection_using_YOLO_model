# PCB Defect Detection with YOLOv8 and Transfer Learning

## Overview
Welcome to the **PCB Defect Detection Project**! This project aims to automate the process of identifying defects in printed circuit boards (PCBs) using advanced machine learning techniques. By leveraging the power of YOLOv8, a state-of-the-art object detection model, and transfer learning, this project enables efficient and accurate detection of various PCB defects that are crucial for ensuring product quality in manufacturing processes.

### Defects Detected
The model can identify the following PCB defects:
- **Crack**
- **Missing Holes**
- **Mouse Bites**
- **Open Circuit**
- **Short**
- **Spur**
- **Spurious Copper**

## Key Features
- **High Accuracy**: Achieved optimal performance using YOLOv8 with transfer learning.
- **Real-Time Detection**: Capable of detecting defects in real-time.
- **Custom Dataset**: Built with a combination of Kaggle datasets and custom-annotated images to improve detection accuracy.
- **Transfer Learning**: Fine-tuned YOLOv8 for PCB defect detection to leverage pre-trained weights and reduce training time.


## How to Use
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/pcb-defect-detection.git
    cd pcb-defect-detection
    ```

2. **Install Dependencies**:
    Install the required libraries by running:
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare Data**:
    Place your dataset in the `data/` directory.

4. **Train the Model**:
    Run the training script to fine-tune the YOLOv8 model on your dataset:
    ```bash
    python train.py
    ```

5. **Run Inference**:
    To test the model and run inference on a new PCB image, use the following command:
    ```bash
    python inference.py --image path_to_image
    ```

6. **Evaluation**:
    Check the evaluation metrics and performance results after training to assess model accuracy.


## Conclusion
This project showcases the potential of AI in automating defect detection for PCB manufacturing, significantly reducing human error and increasing efficiency. With YOLOv8 and transfer learning, this model provides a reliable and fast solution to detect critical PCB defects in real-time.

