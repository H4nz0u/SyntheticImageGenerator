from .base_transform import Transformation
from ..utilities import register_transformation, logger, get_cached_dataframe
from ..image_management import Image, ImgObject
import cv2
import numpy as np

@register_transformation
class Greyscale(Transformation):
    def __init__(self):
        super().__init__()

    def apply(self, obj: ImgObject):
        logger.info('Converting image to greyscale')
        obj.image = cv2.cvtColor(obj.image, cv2.COLOR_BGR2GRAY)
        obj.image = cv2.cvtColor(obj.image, cv2.COLOR_GRAY2BGR)
        
@register_transformation
class RandomGreyscale(Greyscale):
    def __init__(self, probability: float):
        self.probability = probability
        super().__init__()
        
    def apply(self, obj: ImgObject):
        if np.random.rand() < self.probability:
            super().apply(obj)