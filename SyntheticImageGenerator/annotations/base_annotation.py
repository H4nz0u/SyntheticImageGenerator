from typing import Tuple
from pathlib import Path
import numpy as np

class BaseAnnotator:
    def __init__(self, overwrite_classes: dict = {}):
        self.overwrite_classes = overwrite_classes
    
    def append_object(self, bounding_box, class_label: str):
        raise NotImplementedError("Method not implemented")
    
    
    def write_xml(self, output_path: Path, size: tuple[int, ...]):
        raise NotImplementedError("Method not implemented")

    def __str__(self) -> str:
        return self.__class__.__name__
    
    def reset(self):
        raise NotImplementedError("Method not implemented")