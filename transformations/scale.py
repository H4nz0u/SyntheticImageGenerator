from .base_transform import Transformation
from .transformation_registry import register_transformation
from image_management import ImgObject
import cv2
import numpy as np

@register_transformation
class Scale(Transformation):
    def __init__(self, factor):
        self.factor = factor
        
    def apply(self, obj: ImgObject):
        image = obj.image
        new_width = int(image.shape[1] * self.factor)
        new_height = int(image.shape[0] * self.factor)
        
        new_dim = (new_width, new_height)
        resized_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
        obj.mask = cv2.resize(obj.mask, new_dim, interpolation=cv2.INTER_AREA)
        obj.image = resized_image
        segmentation = obj.segmentation.astype(np.float32)
        segmentation *= np.array(self.factor, dtype=np.float32)
        obj.segmentation = segmentation.astype(np.int32)

        obj.bbox.coordinates = tuple(np.array(obj.bbox.coordinates) * self.factor)

@register_transformation
class RandomScale(Scale):
    def __init__(self, min_factor, max_factor):
        self.min_factor = min_factor
        self.max_factor = max_factor
        super().__init__(np.random.uniform(min_factor, max_factor))
        
    def apply(self, obj: ImgObject):
        super().apply(obj)
        self.factor = np.random.uniform(self.min_factor, self.max_factor)