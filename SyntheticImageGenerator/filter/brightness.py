from .base_filter import Filter
from ..utilities import register_filter, get_cached_dataframe, logger
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

    def apply(self, image: np.ndarray, mask=None):
        """
        Applies the target brightness filter to the image based on the mask.

        :param image: Input image.
        :param mask: Boolean mask indicating object pixels.
        :return: Brightness adjusted image.
        """
        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if mask is not None:
            object_pixels = image_bw[mask == 255]
            current_brightness = np.mean(object_pixels) if object_pixels.size > 0 else np.nan
        else:
            current_brightness = np.mean(image_bw)

        
        if np.isnan(current_brightness) or current_brightness == 0 or self.target_brightness == 0:
            scaling_factor = 1.0
        else:
            scaling_factor = self.target_brightness / current_brightness
        image = cv2.convertScaleAbs(image, alpha=scaling_factor, beta=0)
        return image
    
@register_filter
class RandomTargetBrightness(TargetBrightness):
    def __init__(self, min_brightness: float, max_brightness: float) -> None:
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        random = np.random.uniform(min_brightness, max_brightness)
        super().__init__(random)
    def apply(self, image, mask=None):
        random = np.random.uniform(self.min_brightness, self.max_brightness)
        super().__init__(random)
        return super().apply(image, mask)

@register_filter
class TargetBrightnessFromDataFrame(TargetBrightness):
    def __init__(self, dataframe_path, class_name, column_name="brightness"):
        self.dataframe_path = dataframe_path
        self.column_name = column_name
        self.data = get_cached_dataframe(self.dataframe_path)
        self.cls = class_name
        target_brightness = 1
        super().__init__(target_brightness)

    def _select_target_brightness(self, cls):
        try:
            brightness_value = self.data.sample_parameter(self.column_name, filters={'class': cls})
            return brightness_value
        except Exception as e:
            logger.error(f"Failed to select an brightness value for class '{cls}': {e}")
            raise

    def apply(self, image, mask=None):
        self.target_brightness = self._select_target_brightness(self.cls)
        logger.info(f'Applying Brightness filter using factor from DataFrame: {self.target_brightness}')
        return super().apply(image, mask)