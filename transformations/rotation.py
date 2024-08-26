from .base_transform import Transformation
from utilities import register_transformation, logger, get_cached_dataframe
from image_management import Image, ImgObject
import cv2
import numpy as np

@register_transformation
class Rotate(Transformation):
    def __init__(self, angle):
        self.angle = angle

    def apply(self, obj: ImgObject):
        logger.info(f'Rotating image by {self.angle} degrees')
        image = obj.image
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        
        height, width = image.shape[:2]
        M = cv2.getRotationMatrix2D(image_center, self.angle, 1.0)
        
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        M[0, 2] += (new_width / 2) - image_center[0]
        M[1, 2] += (new_height / 2) - image_center[1]

        image = cv2.warpAffine(image, M, (new_width, new_height))

        obj.image = image
        if obj.segmentation.size > 0:
            self._transform_segmentation(obj, M, new_width, new_height)
        self._transform_bbox(obj, M)
        logger.info(f'New BBox coordinates: {obj.bbox.coordinates}')
        if obj.mask.size > 0:
            self._transform_mask(obj, M, new_height, new_width)

    

    def _transform_mask(self, obj, M, new_height, new_width):
        mask = obj.mask
        # Ensure the mask is a single channel image
        if len(mask.shape) == 3 and mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Perform the affine transformation (rotation) on the mask
        rotated_mask = cv2.warpAffine(mask, M, (new_width, new_height))
        obj.mask = rotated_mask
        
    def _transform_segmentation(self, obj, M, new_width, new_height):
        # Reshape the segmentation array to the appropriate format
        segmentation = np.array(obj.segmentation, dtype=np.float32).reshape((-1, 1, 2))
        
        # Apply the transformation
        transformed_segmentation = cv2.transform(segmentation, M).reshape((-1, 2))
        
        # Clip the transformed segmentation to be within the image bounds
        transformed_segmentation[:, 0] = np.clip(transformed_segmentation[:, 0], 0, new_width)
        transformed_segmentation[:, 1] = np.clip(transformed_segmentation[:, 1], 0, new_height)

        # Update the object's segmentation
        obj.segmentation = transformed_segmentation.astype(np.int32)
        
    def _transform_bbox(self, obj, M):
        obj.bbox.coordinates = self.update_bbox_from_mask(obj.mask)

@register_transformation
class RandomRotate(Rotate):
    def __init__(self, min_angle, max_angle):
        self.min_angle = min_angle
        self.max_angle = max_angle
        super().__init__(np.random.randint(self.min_angle, self.max_angle))
        
    def apply(self, obj: ImgObject):
        super().apply(obj)
        self.angle = np.random.randint(self.min_angle, self.max_angle)
        
@register_transformation
class RotateFromDataFrame(Rotate):
    def __init__(self, dataframe_path, column_name="angle"):
        self.dataframe_path = dataframe_path
        self.column_name = column_name
        self.data = get_cached_dataframe(self.dataframe_path)
        angle = 0
        super().__init__(angle)

    def _select_angle(self, cls):
        try:
            # Filter the DataFrame by the class
            filtered_data = self.data[self.data['class'] == cls]
            angle_values = filtered_data[self.column_name].dropna()
            if len(angle_values) == 0:
                raise ValueError(f"No valid angle values found for class '{cls}' in column '{self.column_name}'.")
            # Sample one angle value from the distribution
            return np.random.choice(angle_values)
        except Exception as e:
            logger.error(f"Failed to select an angle for class '{cls}': {e}")
            raise

    def apply(self, obj: ImgObject):
        self.angle = self._select_angle(obj.cls)
        logger.info(f'Applying rotation using an angle from DataFrame: {self.angle} degrees')
        super().apply(obj)