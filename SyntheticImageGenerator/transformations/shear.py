from .base_transform import Transformation
from ..utilities import register_transformation, logger, get_cached_dataframe
from ..image_management import ImgObject
import cv2
import numpy as np

@register_transformation
class Shear(Transformation):
    def __init__(self, shear_factor, direction='x'):
        """
        Initialize the Shear transformation.

        :param shear_factor: The shear factor to apply.
        :param direction: Direction of shear, either 'x' or 'y'.
        """
        if direction not in ['x', 'y']:
            raise ValueError("Direction must be either 'x' or 'y'.")
        self.shear_factor = shear_factor
        self.direction = direction.lower()

    def apply(self, obj: ImgObject):
        logger.info(f"Applying {self.direction.upper()}-Shear by factor {self.shear_factor}")
        image = obj.image
        height, width = image.shape[:2]

        if self.direction == 'x':
            new_width = int(width + abs(height * self.shear_factor))
            M = np.array([[1, self.shear_factor, 0], [0, 1, 0]], dtype=np.float32)
            output_size = (new_width, height)
        else:  # direction == 'y'
            new_height = int(height + abs(width * self.shear_factor))
            M = np.array([[1, 0, 0], [self.shear_factor, 1, 0]], dtype=np.float32)
            output_size = (width, new_height)

        # Apply shear to image
        transformed_image = cv2.warpAffine(image, M, output_size)
        obj.image = transformed_image

        # Apply shear to mask if it exists
        if obj.mask is not None and obj.mask.size > 0:
            transformed_mask = cv2.warpAffine(obj.mask, M, output_size)
            obj.mask = transformed_mask

        # Update bounding box
        obj.bbox.coordinates = self.transform_bbox(obj)

        # Update segmentation if it exists
        if obj.segmentation.size > 0:
            obj.segmentation = self.transform_segmentation(obj.segmentation, M)

        logger.info(f'New BBox coordinates: {obj.bbox.coordinates}')

    def transform_bbox(self, obj):
        return self.update_bbox_from_mask(obj.mask)

    def transform_segmentation(self, segmentation, M):
        points = np.array(segmentation, dtype=np.float32).reshape(-1, 1, 2)
        transformed_points = cv2.transform(points, M)
        transformed_segmentation = transformed_points.reshape(-1, 2)
        return transformed_segmentation


@register_transformation
class RandomShear(Shear):
    def __init__(self, min_shear_factor, max_shear_factor, direction='x'):
        """
        Initialize the RandomShear transformation.

        :param min_shear_factor: Minimum shear factor.
        :param max_shear_factor: Maximum shear factor.
        :param direction: Direction of shear, either 'x' or 'y'.
        """
        if min_shear_factor > max_shear_factor:
            raise ValueError("min_shear_factor must be <= max_shear_factor.")
        self.min_shear_factor = min_shear_factor
        self.max_shear_factor = max_shear_factor
        shear_factor = np.random.uniform(min_shear_factor, max_shear_factor)
        super().__init__(shear_factor, direction)

    def apply(self, obj: ImgObject):
        # Update shear_factor before applying
        self.shear_factor = np.random.uniform(self.min_shear_factor, self.max_shear_factor)
        logger.info(f"Randomizing shear factor to {self.shear_factor}")
        super().apply(obj)


@register_transformation
class ShearFromDataFrame(Shear):
    def __init__(self, dataframe_path, column_name="shear", direction='x'):
        """
        Initialize the ShearFromDataFrame transformation.

        :param dataframe_path: Path to the DataFrame containing shear factors.
        :param column_name: Column name in the DataFrame to retrieve shear factors.
        :param direction: Direction of shear, either 'x' or 'y'.
        """
        self.dataframe_path = dataframe_path
        self.column_name = column_name
        self.data = get_cached_dataframe(self.dataframe_path)
        # Initialize with a default shear_factor; it will be updated in apply()
        shear_factor = 1
        super().__init__(shear_factor, direction)

    def _select_shear_factor(self, cls):
        try:
            # Filter the DataFrame by the class
            shear_factor = self.data.sample_parameter(cls, {"class": cls})
            return shear_factor
        except Exception as e:
            logger.error(f"Failed to select a shear factor for class '{cls}': {e}")
            raise

    def apply(self, obj: ImgObject):
        self.shear_factor = self._select_shear_factor(obj.cls)
        logger.info(f'Applying {self.direction.upper()}-Shear using factor from DataFrame: {self.shear_factor}')
        super().apply(obj)
