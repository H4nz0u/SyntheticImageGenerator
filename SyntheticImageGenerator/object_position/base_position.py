from typing import Tuple
class BasePositionDeterminer:
    def __init__(self):
        pass
    def get_position(self, image, objects) -> Tuple[int, int, int, int]:
        raise NotImplementedError('The get_position method must be implemented by the subclass')
    
    def __str__(self) -> str:
        return self.__class__.__name__
    
    def get_absolute_position(self, background, foreground, position_x, position_y) -> Tuple[int, int, int, int]:
        obj_h, obj_w = foreground.image.shape[:2]
        background_h, background_w = background.shape[:2]
        
        # Compute starting positions based on background dimensions
        x_start = int(background_w * position_x)
        y_start = int(background_h * position_y)
        
        # Adjust positions to ensure the object fits within the background
        x_start = max(min(x_start, background_w - obj_w), 0)
        y_start = max(min(y_start, background_h - obj_h), 0)
        
        x_end = x_start + obj_w
        y_end = y_start + obj_h
        return x_start, y_start, x_end, y_end