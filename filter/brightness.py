from filter import Filter
from utilities import register_filter, get_cached_dataframe, logger
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
            # Filter the DataFrame by the class
            filtered_data = self.data[self.data['class'] == cls]
            brightness_values = filtered_data[self.column_name].dropna()
            if len(brightness_values) == 0:
                raise ValueError(f"No valid brightness values found for class '{cls}' in column '{self.column_name}'.")
            return np.random.choice(brightness_values)
        except Exception as e:
            logger.error(f"Failed to select an brightness value for class '{cls}': {e}")
            raise

    def apply(self, img):
        self.target_brightness = self._select_target_brightness(self.cls)
        logger.info(f'Applying Brightness filter using factor from DataFrame: {self.target_brightness}')
        return super().apply(img)