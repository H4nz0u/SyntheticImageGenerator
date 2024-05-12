from .base_transform import Transformation
from .transformation_registry import register_transformation
from image_management import Image, Object
import cv2
import numpy as np

@register_transformation
class Rotate(Transformation):
    def __init__(self, angle):
        self.angle = angle
    """
    def __init__(self, min_angle, max_angle):
        self.angle = np.random.randint(min_angle, max_angle)
    """
    def apply(self, obj: Object):
        image = obj.get_np_image()
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        
        height, width = image.shape[:2]
        M = cv2.getRotationMatrix2D(image_center, self.angle, 1.0)
        
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        M[0, 2] += (new_width / 2) - image_center[0]
        M[1, 2] += (new_height / 2) - image_center[1]
        
        image = cv2.warpAffine(image, M, (new_width, new_height))
        return image