from .base_transform import Transformation
from utilities import register_transformation
from image_management import ImgObject
import cv2
import numpy as np

@register_transformation
class Shear(Transformation):
    def __init__(self, shear_factor):
        self.shear_factor = shear_factor
    def apply(self, obj: ImgObject):
        image = obj.image
        height, width =  image.shape[:2]
        M = np.float32([[1, self.shear_factor, 0], [0, 1, 0]]) 
        image = cv2.warpAffine(image, M, (int(width+height*self.shear_factor), height))
        obj.image = image
        obj.mask = cv2.warpAffine(obj.mask, M, (int(width+height*self.shear_factor), height))
        obj.bbox.coordinates = self.transform_bbox(obj.bbox.coordinates, M, height)
        obj.segmentation = self.transform_segmentation(obj.segmentation, M)
    
    def transform_bbox(self, bbox_coords, M, height):
        x, y, w, h = bbox_coords
        corners = np.array([
            [x, y],          
            [x + w, y + h] 
        ])

        transformed_corners = cv2.transform(np.array([corners], dtype=np.float32), M)[0]

        new_x = int(transformed_corners[0][0])
        new_y = int(transformed_corners[0][1])
        new_w = int(abs(transformed_corners[1][0] - transformed_corners[0][0]))
        new_h = h 

        return np.array([new_x, new_y, new_w, new_h])
    
    def transform_segmentation(self, segmentation, M):
        points = np.array(segmentation, dtype=np.float32).reshape(-1, 1, 2)
        
        transformed_points = cv2.transform(points, M)
        
        transformed_segmentation = transformed_points.reshape(-1, 2)
        
        return transformed_segmentation


@register_transformation
class RandomShear(Shear):
    def __init__(self, min_shear_factor, max_shear_factor):
        self.min_shear_factor = min_shear_factor
        self.max_shear_factor = max_shear_factor
        super().__init__(np.random.uniform(min_shear_factor, max_shear_factor))

    def apply(self, image: cv2.typing.MatLike):
        super().apply(image)
        self.shear_factor = np.random.uniform(self.min_shear_factor, self.max_shear_factor)
