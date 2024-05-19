from typing import Tuple
class BaseAnnotator:
    def __init__(self):
        pass
    
    def append_object(self, bounding_box, class_label: str):
        raise NotImplementedError("Method not implemented")
    
    def write_xml(self, output_path: str):
        raise NotImplementedError("Method not implemented")

    def __str__(self) -> str:
        return self.__class__.__name__