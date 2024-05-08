from .base_transform import Transformation
from .transformation_registry import register_transformation
from image_management.image import Image
import cv2
import numpy as np

@register_transformation
class Scale(Transformation):
    def __init__(self, scale):
        self.scale = scale
        
    def apply(self, image: cv2.typing.MatLike):
        new_width = int(image.shape[1] * self.scale)
        new_height = int(image.shape[0] * self.scale)
        
        new_dim = (new_width, new_height)
        resized_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
        return resized_image