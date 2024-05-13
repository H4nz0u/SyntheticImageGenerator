import cv2
from image_management.object import ImgObject
import numpy as np
class Scene:
    def __init__(self, background, foregrounds) -> None:
        self.background = background
        self.foregrounds = foregrounds
        self.filters = []
    
    def add_filter(self, filter):
        self.filters.append(filter)
    
    def apply_filter(self):
        for filter in self.filters:
            filter.apply(self.background)
            
    def add_foreground(self, foreground: ImgObject):
        position_x, position_y = 0.5, 0.5  # Center position by default
        obj_h, obj_w = foreground.image.shape[:2]
        background_h, background_w = self.background.shape[:2]

        # Calculate the center position of the foreground on the background
        x_start = int((background_w - obj_w) * position_x)
        y_start = int((background_h - obj_h) * position_y)

        # Clamping the start values to ensure they are within the background dimensions
        x_start = max(min(x_start, background_w - obj_w), 0)
        y_start = max(min(y_start, background_h - obj_h), 0)

        # Calculate the end points, ensuring they do not exceed the background dimensions
        x_end = min(x_start + obj_w, background_w)
        y_end = min(y_start + obj_h, background_h)

        # Adjusting the foreground image and mask dimensions if necessary
        cropped_foreground_width = x_end - x_start
        cropped_foreground_height = y_end - y_start

        # Resizing the mask and converting it to ensure it matches the cropped foreground dimensions
        mask = cv2.GaussianBlur(foreground.mask, (21, 21), 0)
        mask = cv2.resize(mask, (cropped_foreground_width, cropped_foreground_height))
        
        mask = mask.astype(np.float32) / 255.0  # Normalize mask to range 0-1 for blending
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if len(mask.shape) == 2 else mask
        
        # Crop the foreground image to the exact size of the region
        foreground_region = foreground.image[:cropped_foreground_height, :cropped_foreground_width]
        # Blend the foreground onto the background
        roi = self.background[y_start:y_end, x_start:x_end]
        self.background[y_start:y_end, x_start:x_end] = roi * (1 - mask) + foreground_region * mask



        # Draw bounding box on the final image for testing
        """self.image = cv2.polylines(
            car_image_with_sign, [self.signImage.bounding_box], True, (0, 255, 0), 2
        )"""
