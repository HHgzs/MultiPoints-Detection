import cv2

from ultralytics import YOLO

# Load the YOLO model
model = YOLO("ultralytics/cfg/models/11/yolo11-multipoints.yaml").load("runs/multipoints/multipoints2/weights/epoch99.pt")

video_path = 'main/test.mp4'
video_name = video_path.split('/')[-1].split('.')[0]

cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()