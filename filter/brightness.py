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

    def apply(self, image, bbox=None):
        """
        Applies the target brightness filter to the image.

        :param image: Input image.
        :param bbox: Optional bounding box (xmin, ymin, xmax, ymax) to calculate brightness.
        :return: Brightness adjusted image.
        """
        if bbox is not None and len(bbox) == 4:
            xmin, ymin, xmax, ymax = bbox
            #convert to int
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            bbox_image = image[ymin:ymax, xmin:xmax]
            current_brightness = np.mean(bbox_image) if bbox_image.size > 0 else np.nan
        else:
            # Default to the whole image mean if no bbox or invalid bbox is provided
            current_brightness = np.mean(image)

        if np.isnan(current_brightness) or current_brightness == 0 or self.target_brightness == 0:
            scaling_factor = 1
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
    def apply(self, image, bbox=None):
        random = np.random.uniform(self.min_brightness, self.max_brightness)
        super().__init__(random)
        return super().apply(image, bbox)

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

    def apply(self, image, bbox=None):
        self.target_brightness = self._select_target_brightness(self.cls)
        logger.info(f'Applying Brightness filter using factor from DataFrame: {self.target_brightness}')
        return super().apply(image, bbox)