from .image import Image
from typing import List
from transformations.base_transform import BaseTransform
from pathlib import Path
import json
from utilities.boundingbox import BoundingBox
class Object:
    def __init__(self, image_path: Path, annotation_path: Path) -> None:
        self.image = Image(image_path)
        self.annoations = self.load_annotation(annotation_path)
        self.bbox, self.cl = self.parse_annotations(self.annoations)
        
    def load_annotation(self, annotation_path: Path):
        with open(annotation_path, 'r') as file:
            data = json.load(file)
        return data
    
    def parse_annotations(annotation_data):
        item = annotation_data['objects'][0]
        bbox = BoundingBox(
            coordinates=(item['bbox']['x'], item['bbox']['y'], item['bbox']['width'], item['bbox']['height']),
            format_type="min_max"
        )
        return bbox, item["class"]
        
    def apply_transformations(self, transformations: List[BaseTransform]):
        for transformation in transformations:
            transformation.apply(self.image.get_image())