from ultralytics import YOLO
import cv2


model = YOLO('robomasters/yolov8_train2/weights/best.pt')  # Model weights

# Image paths to validate
image_paths = [
    "C:\\Users\\cjvan\\OneDrive\\Desktop\\Repository\\robomasters.v1i.yolov11\\valid\\images\\frame-449_jpg.rf.052dfc31fd94f3b866781f33d1f93604.jpg"
]

# Run inference on each image
for image_path in image_paths:
    # Run inference
    results = model.predict(image_path, save=True, conf=0.25, imgsz=640)

    # Display the annotated image
    for result in results:
        annotated_frame = result.plot()  # Get the annotated frame
        cv2.imshow("Annotated Image", annotated_frame)
        cv2.waitKey(0)  # Wait for a key press to move to the next image

# Close all OpenCV windows
cv2.destroyAllWindows()