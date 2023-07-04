import argparse
import cv2
from video_processing import VideoProcessor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply camera shake effect to a video')
    parser.add_argument('--input_video', type=str, help='Path to the input video file')
    parser.add_argument('--output_video', type=str, help='Path to the output video file')
    args = parser.parse_args()

    border_size = 0
    smoothing_window = 150

    # Check if input and output video paths are provided
    input_video = "data/06.mp4"
    if args.input_video:
        input_video = args.input_video
    # Initialize video capture from webcam
    cap = cv2.VideoCapture(input_video)

    # Initialize video writer for the output video
    if args.output_video:
        output_path = args.output_video
    else:
        output_path = 'output/output_video.mp4'
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize video processor
    video_processor = VideoProcessor()

    cv2.namedWindow("Original")

    while True:
        # Read frame from video
        ret, frame = cap.read()
        if not ret:
            break

        # Apply camera shake effect
        processed_frame = video_processor.apply_camera_shake(frame, border_size, smoothing_window)

        # Display original and processed video frames
        cv2.imshow('Original', frame)
        cv2.imshow('Processed', processed_frame)

        # Write frame to the output video
        out.write(processed_frame)

        # Check for key press
        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' to exit
            break

    # Release video capture and writer
    cap.release()
    out.release()

    # Close all windows
    cv2.destroyAllWindows()
