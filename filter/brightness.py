from filter import Filter
from utilities import register_filter
import cv2
import numpy as np

@register_filter
class Brightness(Filter):
    def __init__(self, brightness_factor: float) -> None:
        self.brightness_factor = brightness_factor
    def apply(self, image):
        image = cv2.convertScaleAbs(image, alpha=self.brightness_factor, beta=0)
        return image

@register_filter
class TargetBrightness(Filter):
    def __init__(self, target_brightness: float) -> None:
        self.target_brightness = target_brightness

    def apply(self, image):
        current_brightness = np.mean(image)

        if current_brightness == 0:
            scaling_factor = 0
        else:
            scaling_factor = self.target_brightness / current_brightness

        image = cv2.convertScaleAbs(image, alpha=scaling_factor, beta=0)

        return image

@register_filter
class RandomTargetBrightness(TargetBrightness):
    def __init__(self, min_brightness: float, max_brightness: float) -> None:
        super().__init__(np.random.randint(min_brightness, max_brightness))