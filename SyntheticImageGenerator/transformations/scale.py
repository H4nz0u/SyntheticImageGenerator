from .base_transform import Transformation
from ..utilities import register_transformation, logger, get_cached_dataframe
from ..image_management import ImgObject
import cv2
import numpy as np

@register_transformation
class Scale(Transformation):
    def __init__(self, factor):
        self.factor = factor
        
    def apply(self, obj: ImgObject):
        logger.info(f'Scaling image by {self.factor}')
        image = obj.image
        image_height, image_width = image.shape[:2]
        new_width = int(image.shape[1] * self.factor)
        new_height = int(image.shape[0] * self.factor)
        
        new_dim = (new_width, new_height)
        resized_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
        if obj.mask is not None:
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
class ScaleToArea(Transformation):
    def __init__(self, target_area_ratio: float, background_size: int = 1127000):
        """
        Initializes the ScaleToArea transformation.

        Args:
            target_area_ratio (float): Desired ratio of the BBox area to the background size (e.g., 0.1 for 10%).
            background_size (int, optional): Reference area (e.g., total image area). Defaults to 1127000.
        """
        if not (0 < target_area_ratio <= 1):
            raise ValueError("target_area_ratio must be between 0 (exclusive) and 1 (inclusive).")
        if background_size <= 0:
            raise ValueError("background_size must be a positive integer.")
        
        self.target_area_ratio = target_area_ratio
        self.background_size = background_size

    def apply(self, obj: ImgObject):
        """
        Applies scaling to the image such that the BBox area occupies the target_area_ratio of the background_size.

        This is done in two steps:
        1. Calculate the scaling factor for the BBox area.
        2. Calculate and apply the scaling factor for the entire image based on the BBox scaling factor.

        Args:
            obj (ImgObject): The image object containing image, mask, segmentation, and bbox.
        """
        logger.info("Starting ScaleToArea transformation.")

        # Extract the current image and BBox area
        image = obj.image
        current_bbox_area = obj.bbox.area()

        if current_bbox_area <= 0:
            logger.warning("BBox area is zero or negative. Skipping scaling.")
            return

        # Step 1: Calculate the desired BBox area based on target_area_ratio
        desired_bbox_area = self.target_area_ratio * self.background_size

        # Step 2: Calculate the scaling factor for the BBox area (f_bbox)
        # f_bbox = desired_bbox_area / current_bbox_area
        f_bbox = desired_bbox_area / current_bbox_area

        if f_bbox <= 0:
            logger.warning("Calculated BBox scaling factor is non-positive. Skipping scaling.")
            return

        # Step 3: Calculate the image scaling factor (f_image) based on f_bbox
        # Since area scales with the square of the scaling factor, f_image = sqrt(f_bbox)
        f_image = np.sqrt(f_bbox)
        logger.info(f"Calculated image scaling factor (f_image): {f_image}")

        # Validate the scaling factor
        if f_image <= 0:
            logger.warning("Calculated image scaling factor is non-positive. Skipping scaling.")
            return

        # Step 4: Apply the scaling factor to the entire image
        image_height, image_width = image.shape[:2]
        new_width = max(1, int(image_width * f_image))  # Ensure dimensions are at least 1
        new_height = max(1, int(image_height * f_image))
        new_dim = (new_width, new_height)
        logger.info(f"Original image size: ({image_width}, {image_height})")
        logger.info(f"New image size: {new_dim}")

        # Resize the image
        resized_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
        obj.image = resized_image

        # Resize the mask if it exists
        if obj.mask is not None and obj.mask.size > 0:
            resized_mask = cv2.resize(obj.mask, new_dim, interpolation=cv2.INTER_AREA)
            obj.mask = resized_mask
            logger.debug("Mask resized.")

        if obj.segmentation.size > 0:
            segmentation = obj.segmentation.astype(np.float32)
            segmentation *= np.array(f_image, dtype=np.float32)
            obj.segmentation = segmentation.astype(np.int32)
            logger.debug("Segmentation resized.")

        # Step 5: Scale the BBox coordinates
        original_coordinates = np.array(obj.bbox.coordinates)
        scaled_coordinates = original_coordinates * f_image
        obj.bbox.coordinates = scaled_coordinates

        # Optionally, update the BBox area after scaling
        new_bbox_area = obj.bbox.area()
        logger.info("ScaleToArea transformation completed successfully.")

            
@register_transformation
class RandomScaleToArea(ScaleToArea):
    def __init__(self, min_target_area_ratio: float, max_target_area_ratio: float, background_size: int  = 1):
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
            scale_factor = self.data.sample_parameter(self.column_name, filters={'class': cls})
            return scale_factor
        except Exception as e:
            logger.error(f"Failed to select an factor for class '{cls}': {e}")
            raise

    def apply(self, obj: ImgObject):
        self.factor = self._select_factor(obj.cls)
        logger.info(f'Applying Scaling using factor from DataFrame: {self.factor}')
        super().apply(obj)

@register_transformation
class ScaleToAreaFromDataFrame(ScaleToArea):
    def __init__(self, dataframe_path, background_size: int =0, column_name="area"):
        self.dataframe_path = dataframe_path
        self.column_name = column_name
        self.data = get_cached_dataframe(self.dataframe_path)
        target_area_ratio = 1
        if background_size > 0:
            super().__init__(target_area_ratio, background_size)
        else:
            super().__init__(target_area_ratio)

    def _select_target_area_ratio(self, cls):
        try:
            scale_factor = self.data.sample_parameter(self.column_name, filters={'class': cls})
            return scale_factor
        except Exception as e:
            logger.error(f"Failed to select an ratio for class '{cls}': {e}")
            raise

    def apply(self, obj: ImgObject):
        self.target_area_ratio = self._select_target_area_ratio(obj.cls)
        logger.info(f'Applying Scaling using ratio from DataFrame: {self.target_area_ratio}')
        super().apply(obj)