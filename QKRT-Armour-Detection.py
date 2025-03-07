from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO('robomasters/yolov8_train2/weights/best.pt')  # Path to your trained model

# index 0 is webcam 
cap = cv2.VideoCapture(0)

# Check camera 
if not cap.isOpened():
    print("Error: Could not open built-in camera.")
    exit()

# Process frame from the camera
while True:
    ret, frame = cap.read()

    # Frame check
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run inference on the frame
    results = model.track(frame, show=False)  # Set show=False to handle display manually

    
    annotated_frame = results[0].plot()  
    cv2.imshow("YOLOv8 Built-in Camera Feed", annotated_frame)

    # Use Q to quit 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release
cap.release()
cv2.destroyAllWindows()