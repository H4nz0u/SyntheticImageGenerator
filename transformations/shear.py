from .base_transform import Transformation
from .transformation_registry import register_transformation
from image_management.image import Image
import cv2
import numpy as np

@register_transformation
class Shear(Transformation):
    def __init__(self, shear_factor):
        self.shear_factor = shear_factor
    def apply(self, image: cv2.typing.MatLike):
        height, width =  image.shape[:2]
        M = np.float32([[1, self.shear_factor, 0], [0, 1, 0]]) 
        image = cv2.warpAffine(image, M, (width+height*self.shear_factor, height))
        return image
    
@register_transformation
class RandomShear(Transformation):
    def __init__(self, min_shear_factor, max_shear_factor):
        self.min_shear_factor = min_shear_factor
        self.max_shear_factor = max_shear_factor
    def apply(self, image: cv2.typing.MatLike):
        shear_factor = np.random.uniform(self.min_shear_factor, self.max_shear_factor)
        height, width =  image.shape[:2]
        M = np.float32([[1, shear_factor, 0], [0, 1, 0]]) 
        image = cv2.warpAffine(image, M, (width+height*shear_factor, height))
        return image