import numpy as np
import cv2
from typing import List
from pathlib import Path
from utilities.boundingbox import BoundingBox
from utilities import parse_voc_xml

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
        self.segmentation: np.array = data['segmentation']
        self.cls = data['class']
    
    def cut_out_object(self):
        x, y, w, h = self.bbox.coordinates
        self.image = self.image[int(y):int(y)+int(h), int(x):int(x+w)]
        self.segmentation -= np.array([x, y], dtype=np.int32)

        # Create the mask only once
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [self.segmentation], -1, 255, thickness=cv2.FILLED)


        """
        # Combine the mask creation
        black_mask = np.all(self.image == [0, 0, 0], axis=-1)
        self.image[black_mask] = (0, 255, 0)

        black_mask = np.all(self.image == [0, 255, 0], axis=-1)
        self.image[black_mask] = (0, 0, 0)
        """
        # Apply the mask to the image
        self.image = cv2.bitwise_and(self.image, self.image, mask=mask)
        self.mask = mask
        self.bbox.coordinates = (0, 0, w, h)
        
    def apply_transformations(self, transformations: List):
        for transformation in transformations:
            transformation.apply(self)
