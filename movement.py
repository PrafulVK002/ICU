import cv2
import os
import logging
from ultralytics import YOLO


# Function to print alert message
def print_alert():
    print("An unusual moment has been detected!")


# Suppress the ultralytics logger
logging.getLogger("ultralytics").setLevel(logging.ERROR)


# Function to plot bounding boxes without confidence values
def plot_boxes(frame, boxes, class_names):
    for box in boxes:
        cls_id = int(box.cls.item())
        label = f"{class_names[cls_id]}"
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put the label above the bounding box
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


# Main function for video analysis and alerting
def detect_unusual_moments(video_path, model, class_id=1):
    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Initialize background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply background subtraction to detect motion
        fgmask = fgbg.apply(frame)

        # Count number of white pixels (indicating motion)
        motion_pixels = cv2.countNonZero(fgmask)

        # Set threshold for detecting unusual moment (adjust as needed)
        threshold = 5000  # Example threshold value

        #if motion_pixels > threshold:
            # Unusual moment detected, print alert
         #   print_alert()
            # Additional actions can be added here, such as saving frames or recording video

        # Run YOLOv8 inference on the frame
        results = model.predict(frame, stream=True, conf=0.25)  # Lowered confidence threshold to 0.25

        # Check for the presence of the specified class (patient)
        patient_detected = False
        for result in results:
            if any(box.cls.item() == class_id for box in result.boxes):
                patient_detected = True
                print("Patient detected!")

            # Visualize the results on the frame
            plot_boxes(frame, result.boxes, model.names)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", frame)

        if patient_detected:
            print_alert()

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# Path to the pre-trained YOLOv8 model
model_path = os.path.join('.', 'runs', 'detect', 'train13', 'weights', 'last.pt')

# Load YOLOv8 model
model = YOLO(model_path)

# Example usage
VIDEOS_DIR = os.path.join('.', 'testvideo')
video_path = os.path.join(VIDEOS_DIR, 'datavideo_002.mp4')  # Update with your video file path
detect_unusual_moments(video_path, model, class_id=1)
