from .base_transform import Transformation
from .transformation_registry import register_transformation
from image_management import Image, ImgObject
import cv2
import numpy as np

@register_transformation
class Rotate(Transformation):
    def __init__(self, angle):
        self.angle = angle

    def apply(self, obj: ImgObject):
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
        self._transform_segmentation(obj, M, new_width, new_height)
        self._transform_bbox(obj, M)
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
        x, y, w, h = obj.bbox.coordinates
        corners = np.array([
            [x, y],
            [x + w, y],
            [x, y + h],
            [x + w, y + h]
        ])
        corners_homogeneous = np.hstack([corners, np.ones((4, 1))])
        transformed_corners = M @ corners_homogeneous.T
        new_x = np.min(transformed_corners[0])
        new_y = np.min(transformed_corners[1])
        new_w = np.max(transformed_corners[0]) - new_x
        new_h = np.max(transformed_corners[1]) - new_y
        obj.bbox.coordinates = (new_x, new_y, new_w, new_h)

@register_transformation
class RandomRotate(Rotate):
    def __init__(self, min_angle, max_angle):
        self.min_angle = min_angle
        self.max_angle = max_angle
        super().__init__(np.random.randint(self.min_angle, self.max_angle))
        
    def apply(self, obj: ImgObject):
        super().apply(obj)
        self.angle = np.random.randint(self.min_angle, self.max_angle)