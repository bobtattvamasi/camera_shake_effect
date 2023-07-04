'''
video_processing.py: Модуль, содержащий функции и классы для обработки видео.
'''

import cv2
import numpy as np
from vidstab import VidStab
from video_inpainting import VideoInpainting
import traceback


class VideoProcessor:
    def __init__(self):
        self.stabilizer = VidStab()
        self.video_inpainter = VideoInpainting('weights/trained_model_128_149_0.0.pth')

    def apply_camera_shake(self, frame, border_size=0, smoothing_window=30):
        """
                Applies camera shake effect to the input frame.

                Args:
                    frame (numpy.ndarray): The input frame to apply camera shake to.
                    border_size (int): The size of the border to add during stabilization. Default is 0.
                    smoothing_window (int): The size of the window used for smoothing the stabilization. Default is 30.

                Returns:
                    numpy.ndarray: The frame with camera shake effect applied.
        """
        if frame is not None:
            stabilized_frame = self.stabilizer.stabilize_frame(input_frame=frame, smoothing_window=smoothing_window,
                                                               border_size=border_size,
                                                               border_type='black')
        else:
            return None

        center = (stabilized_frame.shape[1] // 2, stabilized_frame.shape[0] // 2)
        radius = min(center[0], center[1])
        output_frame = self.extract_circular_region(stabilized_frame, center, radius)
        #cv2.imshow("shaked", output_frame)

        # Apply video inpainting to fill black borders
        inpainted_frame = self.video_inpainter.inpaint_frame(output_frame)
        result = stabilized_frame

        center = (output_frame.shape[1] // 2, output_frame.shape[0] // 2)
        radius = min(center[0], center[1])

        try:
            result = self.fill_black_borders(output_frame, inpainted_frame, center, radius)
        except Exception as e:
            traceback.print_exc()

        return result

    def fill_black_borders(self, stabilized_video, generated_image, center, radius):
        """
                Fills the black borders in the stabilized video with the generated image.

                Args:
                    stabilized_video (numpy.ndarray): The stabilized video frame.
                    generated_image (numpy.ndarray): The generated image to fill the black borders with.
                    center (tuple): The center coordinates of the circular region.
                    radius (int): The radius of the circular region.

                Returns:
                    numpy.ndarray: The stabilized video with black borders filled using the generated image.
        """
        # Create a mask for the black borders
        mask = np.all(stabilized_video == [0, 0, 0], axis=-1)
        mask_resized = cv2.resize(mask.astype(np.uint8), (generated_image.shape[1], generated_image.shape[0]))

        # Resize the stabilized video to match the shape of the generated image
        stabilized_video_resized = cv2.resize(stabilized_video, (generated_image.shape[1], generated_image.shape[0]))
        #cv2.imshow("stabilized_video_resized", stabilized_video_resized)

        # Apply the mask to combine the stabilized video and the generated image
        result = np.where(mask_resized[..., np.newaxis], generated_image, stabilized_video_resized)
        #cv2.imshow("result", result)

        result = self.extract_circular_region(result, center, radius)

        return result

    @staticmethod
    def extract_circular_region(frame, center, radius):
        """
                Extracts the circular region from the input frame.

                Args:
                    frame (numpy.ndarray): The input frame to extract the circular region from.
                    center (tuple): The center coordinates of the circular region.
                    radius (int): The radius of the circular region.

                Returns:
                    numpy.ndarray: The frame with the circular region extracted.
        """
        # Crop the frame within the circular region
        height, width, _ = frame.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        mask = np.expand_dims(mask, axis=-1)
        masked_frame = np.where(mask == 0, (255, 255, 255), frame)
        masked_frame = masked_frame.astype(np.uint8)
        return masked_frame