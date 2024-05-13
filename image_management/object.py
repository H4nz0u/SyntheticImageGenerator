from .image import Image
from typing import List
from pathlib import Path
import json
from utilities.boundingbox import BoundingBox
from utilities import parse_voc_xml
import numpy as np
import cv2

class ImgObject:
    def __init__(self, image_path: str, annotation_path: str) -> None:
        self.image = cv2.imread(image_path)
        self.parse_annotation(annotation_path)
        self.cut_out_object()
    
    def get_np_image(self):
        return self.image.get_image()
        
    def parse_annotation(self, annotation_path: str):    
       data = parse_voc_xml(annotation_path)
       self.bbox = BoundingBox(
           coordinates=(data['bbox']['x'], data['bbox']['y'], data['bbox']['width'], data['bbox']['height']),
           format_type="min_max"
       )
       self.segmentation = data['segmentation']
       self.cls = data['class']
    
    def cut_out_object(self):
        x, y, w, h = self.bbox.coordinates
        black_mask = np.all(self.image == [0, 0, 0], axis=-1)

        # Replace black pixels with the new color
        self.image[black_mask] = (0, 255, 0)
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [np.array(self.segmentation, dtype=np.int32).reshape((-1, 1, 2))], -1, 255, thickness=cv2.FILLED)
        black_mask = np.all(self.image == [0, 255, 0], axis=-1)

        # Replace black pixels with the new color
        self.image[black_mask] = (0, 0, 0)

        # Apply the mask to the image
        self.image = cv2.bitwise_and(self.image, self.image, mask=mask)
        self.mask = mask[int(y):int(y)+int(h), int(x):int(x+w)]  
        self.image = self.image[int(y):int(y)+int(h), int(x):int(x+w)]
        self.segmentation = [(point[0] - x, point[1] - y) for point in self.segmentation]
        self.bbox.coordinates = (0, 0, w, h)
        
    def apply_transformations(self, transformations: List):
        for transformation in transformations:
            transformation.apply(self.image.get_image())