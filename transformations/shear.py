from .base_transform import Transformation
from utilities import register_transformation
from image_management import ImgObject
import cv2
import numpy as np
from utilities import logger, get_cached_dataframe
@register_transformation
class ShearX(Transformation):
    def __init__(self, shear_factor):
        self.shear_factor = shear_factor
    def apply(self, obj: ImgObject):
        logger.info(f"Applying X-Shear by factor {self.shear_factor}")
        image = obj.image
        height, width =  image.shape[:2]
        
        new_width = int(width + abs(height * self.shear_factor))
        M = np.array([[1, self.shear_factor, 0], [0, 1, 0]], dtype=np.float32) 
        image = cv2.warpAffine(image, M, (new_width, height))
        obj.image = image
        if obj.mask:
            if obj.mask.size > 0:
                obj.mask = cv2.warpAffine(obj.mask, M, (new_width, height))
        obj.bbox.coordinates = self.transform_bbox(obj)
        if obj.segmentation.size > 0:
            obj.segmentation = self.transform_segmentation(obj.segmentation, M)
        logger.info(f'New BBox coordinates: {obj.bbox.coordinates}')

    
    def transform_bbox(self, obj):
        return self.update_bbox_from_mask(obj.mask)
        """x, y, w, h = bbox_coords
        corners = np.array([
            [x, y],                   # Top-left
            [x + w, y],               # Top-right
            [x, y + h],               # Bottom-left
            [x + w, y + h]            # Bottom-right
        ])

        # Apply the transformation matrix to the corners
        transformed_corners = cv2.transform(np.array([corners], dtype=np.float32), M)[0]
        # Recalculate bounding box from the transformed corners
        new_x = min(transformed_corners[:, 0])
        new_y = min(transformed_corners[:, 1])
        new_w = max(transformed_corners[:, 0]) - new_x
        new_h = max(transformed_corners[:, 1]) - new_y
        return np.array([new_x, new_y, new_w, new_h])"""
    
    def transform_segmentation(self, segmentation, M):
        points = np.array(segmentation, dtype=np.float32).reshape(-1, 1, 2)
        
        transformed_points = cv2.transform(points, M)
        
        transformed_segmentation = transformed_points.reshape(-1, 2)
        
        return transformed_segmentation


@register_transformation
class RandomShearX(ShearX):
    def __init__(self, min_shear_factor, max_shear_factor):
        self.min_shear_factor = min_shear_factor
        self.max_shear_factor = max_shear_factor
        super().__init__(np.random.uniform(min_shear_factor, max_shear_factor))

    def apply(self, obj: ImgObject):
        super().apply(obj)
        self.shear_factor = np.random.uniform(self.min_shear_factor, self.max_shear_factor)

@register_transformation
class ShearY(Transformation):
    def __init__(self, shear_factor):
        self.shear_factor = shear_factor

    def apply(self, obj: ImgObject):
        image = obj.image
        height, width = image.shape[:2]
        M = np.array([[1, 0, 0], [self.shear_factor, 1, 0]], dtype=np.float32)
        new_height = int(height + abs(width * self.shear_factor))

        # Apply shear transformation to the image and mask
        obj.image = cv2.warpAffine(image, M, (width, new_height))
        if obj.mask:
            if obj.mask.size > 0:
                obj.mask = cv2.warpAffine(obj.mask, M, (width, new_height))
        
        obj.bbox.coordinates = self.transform_bbox(obj)
        if obj.segmentation.size > 0:
            obj.segmentation = self.transform_segmentation(obj.segmentation, M)
    
    def transform_bbox(self, obj):
        return self.update_bbox_from_mask(obj.mask)
        """
        x, y, w, h = bbox_coords
        corners = np.array([
            [x, y],                   # Top-left
            [x + w, y],               # Top-right
            [x, y + h],               # Bottom-left
            [x + w, y + h]            # Bottom-right
        ])

        # Apply the transformation matrix to the corners
        transformed_corners = cv2.transform(np.array([corners], dtype=np.float32), M)[0]
        # Recalculate bounding box from the transformed corners
        new_x = min(transformed_corners[:, 0])
        new_y = min(transformed_corners[:, 1])
        new_w = max(transformed_corners[:, 0]) - new_x
        new_h = max(transformed_corners[:, 1]) - new_y
        return np.array([new_x, new_y, new_w, new_h])"""
    
    def transform_segmentation(self, segmentation, M):
        points = np.array(segmentation, dtype=np.float32).reshape(-1, 1, 2)
        
        transformed_points = cv2.transform(points, M)
        
        transformed_segmentation = transformed_points.reshape(-1, 2)
        
        return transformed_segmentation

@register_transformation
class RandomShearY(ShearY):
    def __init__(self, min_shear_factor, max_shear_factor):
        self.min_shear_factor = min_shear_factor
        self.max_shear_factor = max_shear_factor
        super().__init__(np.random.uniform(min_shear_factor, max_shear_factor))

    def apply(self, obj: ImgObject):
        super().apply(obj)
        self.shear_factor = np.random.uniform(self.min_shear_factor, self.max_shear_factor)

@register_transformation
class ShearXFromDataFrame(ShearX):
    def __init__(self, dataframe_path, column_name="shear_y"):
        self.dataframe_path = dataframe_path
        self.column_name = column_name
        self.data = get_cached_dataframe(self.dataframe_path)
        shear_factor = 1
        super().__init__(shear_factor)

    def _select_shear_factor(self, cls):
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
        self.shear_factor = self._select_shear_factor(obj.cls)
        logger.info(f'Applying ShearX using factor from DataFrame: {self.shear_factor}')
        super().apply(obj)

@register_transformation
class ShearYFromDataFrame(ShearY):
    def __init__(self, dataframe_path, column_name="shear_y"):
        self.dataframe_path = dataframe_path
        self.column_name = column_name
        self.data = get_cached_dataframe(self.dataframe_path)
        shear_factor = 1
        super().__init__(shear_factor)

    def _select_shear_factor(self, cls):
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
        self.shear_factor = self._select_shear_factor(obj.cls)
        logger.info(f'Applying ShearY using factor from DataFrame: {self.shear_factor}')
        super().apply(obj)