from ..transformations import Transformation
from ..utilities import register_transformation
from ..image_management import ImgObject
from ..utilities import get_cached_dataframe, logger
import numpy as np
import random

@register_transformation
class MatchAspectRatio(Transformation):
    """
    Transformation that crops a random region of the image to match a specified aspect ratio.
    """

    def __init__(self, aspect_ratio: float):
        """
        Initializes the MatchAspectRatio transformation.

        :param aspect_ratio: Desired aspect ratio (width / height) for the cropped region.
        """
        if aspect_ratio <= 0:
            raise ValueError("Aspect ratio must be a positive number.")
        self.aspect_ratio = aspect_ratio

    def apply(self, img: np.ndarray) -> np.ndarray:
        """
        Applies the aspect ratio matching transformation to the image.

        :param obj: ImgObject containing the image to be transformed.
        :return: Transformed image with the specified aspect ratio.
        """
        cropped_image = self._crop_to_aspect_ratio(img)
        return cropped_image

    def _crop_to_aspect_ratio(self, image: np.ndarray) -> np.ndarray:
        """
        Crops the image to the specified aspect ratio at a random position.

        :param image: Original image as a NumPy array.
        :return: Cropped image matching the aspect ratio.
        """
        img_height, img_width = image.shape[:2]
        target_aspect = self.aspect_ratio

        # Calculate target dimensions
        current_aspect = img_width / img_height
        
        #check if they are equal down to 2 decimal places
        if round(current_aspect, 2) == round(target_aspect, 2):
            logger.info("Aspect ratios are equal")
            return image
        
        if current_aspect > target_aspect:
            # Image is wider than target aspect ratio
            new_height = img_height
            new_width = int(target_aspect * new_height)
        else:
            # Image is taller than target aspect ratio
            new_width = img_width
            new_height = int(new_width / target_aspect)

        # Ensure the new dimensions are not larger than the original
        new_width = min(new_width, img_width)
        new_height = min(new_height, img_height)

        # Calculate the maximum top-left corner coordinates for cropping
        max_x = img_width - new_width
        max_y = img_height - new_height

        # Randomly select the top-left corner for cropping
        if max_x > 0:
            x1 = random.randint(0, max_x)
        else:
            x1 = 0

        if max_y > 0:
            y1 = random.randint(0, max_y)
        else:
            y1 = 0

        x2 = x1 + new_width
        y2 = y1 + new_height

        # Perform the cropping
        cropped = image[y1:y2, x1:x2]

        return cropped


@register_transformation
class MatchAspectRatioFromDataFrame(MatchAspectRatio):
    def __init__(self, dataframe_path, column_name="aspect_ratio", class_name=None):
        self.dataframe_path = dataframe_path
        self.column_name = column_name
        self.data = get_cached_dataframe(self.dataframe_path)
        self.aspect_ratio = 1
        self.class_name = class_name
        super().__init__(self.aspect_ratio)
    
    def apply(self, img: np.ndarray) -> np.ndarray:
        self.aspect_ratio = self._select_aspect_ratio(self.class_name)
        return super().apply(img)
    
    def _select_aspect_ratio(self, cls):
        try:
            aspect_ratio = self.data.sample_parameter(self.column_name, filters={"class": cls})
            return aspect_ratio
        except Exception as e:
            logger.error(f"Error selecting aspect ratio for class '{cls}': {e}")
            raise e

@register_transformation
class RandomMatchAspectRatio(MatchAspectRatio):
    def __init__(self, aspect_ratio_range):
        self.aspect_ratio_range = aspect_ratio_range
        super().__init__(1)
    
    def apply(self, img: np.ndarray) -> np.ndarray:
        self.aspect_ratio = self._select_aspect_ratio()
        return super().apply(img)
    
    def _select_aspect_ratio(self):
        try:
            aspect_ratio = random.uniform(*self.aspect_ratio_range)
            return aspect_ratio
        except Exception as e:
            logger.error(f"Error selecting aspect ratio: {e}")
            raise e