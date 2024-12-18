from ..transformations import Transformation
from ..utilities import register_transformation
from ..image_management import ImgObject
import cv2
import numpy as np
import random
from typing import List

@register_transformation
class Zoom(Transformation):
    def __init__(self, zoom_factor, zoom_position_x: float = 0.5, zoom_position_y: float = 0.5):
        self.zoom_factor = zoom_factor
        self.zoom_position: List[float] = [zoom_position_x, zoom_position_y]
        
    def apply(self, obj: ImgObject):
        image = obj.image
        zoom_factor, zoom_window = self.calculate_zoom_parameters(image)
        image = cv2.resize(image[zoom_window[1]:zoom_window[3], zoom_window[0]:zoom_window[2]], (image.shape[1], image.shape[0]))
        return image
    
    def calculate_zoom_parameters(self, image: cv2.typing.MatLike):
        image_height, image_width = image.shape[:2]
        Cx, Cy = image_width // 2, image_height // 2
        Ux, Uy = Cx + (Cx * self.zoom_position[0]), Cy + (Cy * self.zoom_position[0])
        Lx, Ly = Cx - (Cx * self.zoom_position[1]), Cy - (Cy * self.zoom_position[1])
        Px, Py = random.uniform(Lx, Ux), random.uniform(Ly, Uy)
        
        Z = random.uniform(1, self.zoom_factor)
        
        new_width, new_height = int(image_width / Z), int(image_height / Z)
        
        x1, y1 = Px - new_width // 2, Py - new_height // 2
        x2, y2 = Px + new_width // 2, Py + new_height // 2
        
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(image_width, x2)), int(min(image_height, y2))
        
        return Z, (x1, y1, x2, y2)
        
        
        