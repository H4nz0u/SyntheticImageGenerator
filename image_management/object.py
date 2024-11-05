import numpy as np
import cv2
from typing import List
from pathlib import Path
from utilities.boundingbox import BoundingBox
from utilities import parse_voc_xml

class ImgObject:
    bbox: BoundingBox
    image: cv2.typing.MatLike
    mask: cv2.typing.MatLike|None
    def __init__(self, image_path: str, annotation_path: str) -> None:
        try:
            self.image = cv2.imread(image_path)
        except Exception as e:
            print(image_path)
            raise e
        self.parse_annotation(annotation_path)
        self.cut_out_object()
        
    def parse_annotation(self, annotation_path: str):    
        data = parse_voc_xml(annotation_path)
        self.bbox = BoundingBox(
            coordinates=(data['bbox']['x'], data['bbox']['y'], data['bbox']['width'], data['bbox']['height']),
            format_type="min_max"
        )
        segmentation: np.ndarray = data.get("segmentation", None)
        
        if segmentation is None or segmentation.size == 0:
            base = data["bbox"]
            coordinates = [(base["x"], base["y"]), (base["x"], base["y"]+base["height"]), (base["x"]+base["width"], base["y"]+base["height"]), (base["x"]+base["width"], base["y"])]
            self.segmentation = np.array(coordinates)
        else:
            self.segmentation: np.ndarray = segmentation
        self.cls = data['class']
    
    def cut_out_object(self):
        x, y, w, h = self.bbox.coordinates
        self.image = self.image[int(y):int(y)+int(h), int(x):int(x)+int(w)]
        
        if self.segmentation is not None and self.segmentation.size > 0:
            self.segmentation -= np.array([x, y], dtype=np.int32)
            mask: cv2.typing.MatLike = np.zeros(self.image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [self.segmentation], -1, [255], thickness=cv2.FILLED)
            self.image = cv2.bitwise_and(self.image, self.image, mask=mask)
            self.mask = mask
        else:
            self.mask = None  # No mask available if segmentation is not provided
        self.bbox.coordinates = np.array([0, 0, w, h])
        
    def apply_transformations(self, transformations: List):
        for transformation in transformations:
            transformation.apply(self)
