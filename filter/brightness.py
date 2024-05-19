from filter import Filter
from utilities import register_filter
import cv2

@register_filter
class Brightness(Filter):
    def __init__(self, brightness_factor: float) -> None:
        self.brightness_factor = brightness_factor
    def apply(self, image):
        image = cv2.convertScaleAbs(image, alpha=self.brightness_factor, beta=0)
        return image
    