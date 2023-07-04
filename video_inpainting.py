from InpaitingNetwork import InpaintingNetwork
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import Resize
import cv2

class VideoInpainting:
    def __init__(self, model_path):
        self.input_size = 128
        self.model = self.load_model(model_path)
        self.resize_transform_560 = Resize((560, 560))
        self.resize_transform_128 = Resize((self.input_size, self.input_size))


    def load_model(self, model_path):
        # Load the model for video inpainting
        model = InpaintingNetwork(input_size=self.input_size, output_size=self.input_size)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        # Set the model to evaluation mode
        model.eval()

        return model

    def inpaint_frame(self, frame):
        # Apply video inpainting to the frame
        frame_tensor = torch.from_numpy(frame).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
        resized_frame = self.resize_transform_128(frame_tensor)
        inpainted_tensor = self.model(resized_frame)
        inpainted_frame = (inpainted_tensor.squeeze(0).permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
        inpainted_img = Image.fromarray(inpainted_frame)
        inpainted_img = self.resize_transform_560(inpainted_img)

        # Convert PIL Image to OpenCV frame
        inpainted_frame_cv2 = cv2.cvtColor(np.array(inpainted_img), cv2.COLOR_RGB2BGR)

        return inpainted_frame_cv2