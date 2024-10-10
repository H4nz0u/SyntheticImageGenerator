from .base_transform import Transformation
from utilities import register_transformation, logger, get_cached_dataframe
from image_management import ImgObject
import cv2
import numpy as np

@register_transformation
class Scale(Transformation):
    def __init__(self, factor):
        self.factor = factor
        
    def apply(self, obj: ImgObject):
        logger.info(f'Scaling image by {self.factor}')
        image = obj.image
        new_width = int(image.shape[1] * self.factor)
        new_height = int(image.shape[0] * self.factor)
        
        new_dim = (new_width, new_height)
        resized_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
        if obj.mask.size > 0:
            obj.mask = cv2.resize(obj.mask, new_dim, interpolation=cv2.INTER_AREA)
        obj.image = resized_image
        if obj.segmentation.size > 0:
            segmentation = obj.segmentation.astype(np.float32)
            segmentation *= np.array(self.factor, dtype=np.float32)
            obj.segmentation = segmentation.astype(np.int32)
        obj.bbox.coordinates = np.array(obj.bbox.coordinates) * self.factor
        logger.info(f'New BBox coordinates: {obj.bbox.coordinates}')

@register_transformation
class RandomScale(Scale):
    def __init__(self, min_factor, max_factor):
        self.min_factor = min_factor
        self.max_factor = max_factor
        super().__init__(np.random.uniform(min_factor, max_factor))
        
    def apply(self, obj: ImgObject):
        super().apply(obj)
        self.factor = np.random.uniform(self.min_factor, self.max_factor)
        
@register_transformation
class ScaleToArea(Scale):
    def __init__(self, target_area_ratio: float, background_size: int = 1127000):
        self.target_area_ratio = target_area_ratio
        self.background_size = background_size
        
    def apply(self, obj: ImgObject):
        image = obj.image
        obj_area = obj.bbox.area()
        current_area_ratio = obj_area / self.background_size

        if current_area_ratio > 0:
            self.factor = np.sqrt(self.target_area_ratio / current_area_ratio)
            logger.info(f'Scaling object to occupy {self.target_area_ratio * 100}% of the image area (factor: {self.factor})')
            super().apply(obj)
        else:
            logger.warning('Object area is zero, skipping scaling.')
            
@register_transformation
class RandomScaleToArea(ScaleToArea):
    def __init__(self, min_target_area_ratio: float, max_target_area_ratio: float, background_size: int):
        super().__init__(np.random.uniform(min_target_area_ratio, max_target_area_ratio), background_size)

@register_transformation
class ScaleFromDataFrame(Scale):
    def __init__(self, dataframe_path, column_name="area"):
        self.dataframe_path = dataframe_path
        self.column_name = column_name
        self.data = get_cached_dataframe(self.dataframe_path)
        factor = 0
        super().__init__(factor)

    def _select_factor(self, cls):
        try:
            # Filter the DataFrame by the class
            filtered_data = self.data[self.data['class'] == cls]
            factor_values = filtered_data[self.column_name].dropna()
            if len(factor_values) == 0:
                raise ValueError(f"No valid factor values found for class '{cls}' in column '{self.column_name}'.")
            return np.random.choice(factor_values)
        except Exception as e:
            logger.error(f"Failed to select an factor for class '{cls}': {e}")
            raise

    def apply(self, obj: ImgObject):
        self.factor = self._select_factor(obj.cls)
        logger.info(f'Applying Scaling using factor from DataFrame: {self.factor}')
        super().apply(obj)

@register_transformation
class ScaleToAreaFromDataFrame(ScaleToArea):
    def __init__(self, dataframe_path, background_size, column_name="area"):
        self.dataframe_path = dataframe_path
        self.column_name = column_name
        self.data = get_cached_dataframe(self.dataframe_path)
        target_area_ratio = 1
        super().__init__(target_area_ratio, background_size=background_size)

    def _select_target_area_ratio(self, cls):
        try:
            # Filter the DataFrame by the class
            filtered_data = self.data[self.data['class'] == cls]
            ratio_values = filtered_data[self.column_name].dropna()
            if len(ratio_values) == 0:
                raise ValueError(f"No valid ratio values found for class '{cls}' in column '{self.column_name}'.")
            return np.random.choice(ratio_values)
        except Exception as e:
            logger.error(f"Failed to select an ratio for class '{cls}': {e}")
            raise

    def apply(self, obj: ImgObject):
        self.target_area_ratio = self._select_target_area_ratio(obj.cls)
        logger.info(f'Applying Scaling using ratio from DataFrame: {self.target_area_ratio}')
        super().apply(obj)