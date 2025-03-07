from ultralytics import YOLO

# Load a YOLOv8 model
model = YOLO('yolov8n.pt')  # Use YOLOv8 nano model

# Train the model
results = model.train(
    data='C:\\Users\\cjvan\\OneDrive\\Desktop\\Repository\\robomasters.v1i.yolov11\\data.yaml',  # Path to your .yaml file
    
    epochs=100,             # Number of epochs
    batch=16,               # Batch size
    imgsz=640,              # Image size
    device='cpu',           # Use GPU (0) or CPU (leave blank)
    workers=8,              # Number of workers
    
    project='robomasters',  # Project name
    name='yolov8_train'     # Experiment name
)

# Evaluate the model on the validation set
metrics = model.val()

# Save the trained model
model.export(format='onnx')  # Export the model to ONNX format (optional)