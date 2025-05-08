import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("ultralytics/cfg/models/11/yolo11-multipoints.yaml").load("/home/hhgzs/Code/MultiPoints-Detection/main/last.pt")

video_path = '/home/hhgzs/Code/MultiPoints-Detection/main/video/01090809.avi'
video_name = video_path.split('/')[-1].split('.')[0]

cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object
output_path = f'outputs/{video_name}_output.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Write the annotated frame to the output video file
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture object, video writer, and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()