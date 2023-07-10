import cv2
from ai.ai_model import detection, load_yolov5_model

# Load the YOLOv5 model and labels
model, labels = load_yolov5_model()

# Load the video
video_path = "test/video.mp4"
video = cv2.VideoCapture(video_path)

# Get the video's properties (width, height, and frames per second)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
output_path = "test/output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use the appropriate codec for your desired output format
output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    # Read a frame from the video
    ret, frame = video.read()

    if not ret:
        break

    # Perform object detection on the frame
    detected_frame, _, _ = detection(frame, model, labels)

    # Save the detected frame to the output video
    output_video.write(detected_frame)

    # Display the frame
    # cv2.imshow("Video", detected_frame)

    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video and output video, and close all windows
video.release()
output_video.release()
cv2.destroyAllWindows()