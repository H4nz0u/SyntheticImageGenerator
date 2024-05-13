from .base_transform import Transformation
from .transformation_registry import register_transformation
from image_management import Image, ImgObject
import cv2
import numpy as np

@register_transformation
class Rotate(Transformation):
    def __init__(self, angle):
        self.angle = angle
    """
    def __init__(self, min_angle, max_angle):
        self.angle = np.random.randint(min_angle, max_angle)
    """
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
        self._transform_segmentation(obj, M)
        self._transform_bbox(obj, M)
        self._transform_mask(obj, M, new_height, new_width)
    
    def _transform_mask(self, obj, M, new_height, new_width):
        mask = obj.mask
        mask = cv2.warpAffine(mask, M, (new_height, new_width))
        obj.mask = mask
        
    def _transform_segmentation(self, obj, M):
        transformed_segmentation = []
        for point in obj.segmentation:
            # Convert point to homogeneous coordinates
            point_homogeneous = np.array([point[0], point[1], 1])
            # Apply the rotation matrix
            transformed_point = M @ point_homogeneous
            # Store the transformed point, dropping the homogeneous coordinate
            transformed_segmentation.append((transformed_point[0], transformed_point[1]))

        obj.segmentation = transformed_segmentation
    
    def _transform_bbox(self, obj, M):
        # Get the bounding box coordinates
        x, y, w, h = obj.bbox.coordinates
        # Get the four corners of the bounding box
        corners = np.array([
            [x, y],
            [x + w, y],
            [x, y + h],
            [x + w, y + h]
        ])
        # Convert the corners to homogeneous coordinates
        corners_homogeneous = np.hstack([corners, np.ones((4, 1))])
        # Apply the rotation matrix
        transformed_corners = M @ corners_homogeneous.T
        # Get the new bounding box coordinates
        new_x = np.min(transformed_corners[0])
        new_y = np.min(transformed_corners[1])
        new_w = np.max(transformed_corners[0]) - new_x
        new_h = np.max(transformed_corners[1]) - new_y
        # Store the new bounding box coordinates
        obj.bbox.coordinates = (new_x, new_y, new_w, new_h)

@register_transformation
class RandomRotate(Transformation):
    def __init__(self, min_angle, max_angle):
        self.min_angle = min_angle
        self.max_angle = max_angle
        
    def apply(self, obj: ImgObject):
        angle = np.random.randint(self.min_angle, self.max_angle)
        image = obj.image
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        
        height, width = image.shape[:2]
        M = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        M[0, 2] += (new_width / 2) - image_center[0]
        M[1, 2] += (new_height / 2) - image_center[1]
        
        image = cv2.warpAffine(image, M, (new_width, new_height))
        obj.image = image