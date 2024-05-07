from .image import Image
from typing import List
from transformations.base_transform import BaseTransform
class Object:
    def __init__(self, image: Image) -> None:
        self.image = image
    
    def apply_transformations(self, transformations: List[BaseTransform]):
        for transformation in transformations:
            transformation.apply(self.image.get_image())