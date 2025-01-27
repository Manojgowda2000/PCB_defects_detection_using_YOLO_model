from ultralytics import YOLO


model = YOLO(r'G:\pcb_defect_detection\best.pt')

model.train(data=r'G:\pcb_defect_detection\dataset\data.yaml',  # Path to data.yaml
    epochs=50,
    imgsz=640
    # augment=True,
    )