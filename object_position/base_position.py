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
        
        x_start = int((background_w - obj_w) * position_x)
        y_start = int((background_h - obj_h) * position_y)

        x_start = max(min(x_start, background_w - obj_w), 0)
        y_start = max(min(y_start, background_h - obj_h), 0)

        x_end = min(x_start + obj_w, background_w)
        y_end = min(y_start + obj_h, background_h)
        return x_start, y_start, x_end, y_end