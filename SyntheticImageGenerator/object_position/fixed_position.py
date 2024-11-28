from . import BasePositionDeterminer
from ..utilities.registry import register_positionDeterminer

@register_positionDeterminer
class FixedPositionDeterminer(BasePositionDeterminer):
    def __init__(self, x_pos: float, y_pos: float): 
        self.x_pos = x_pos
        self.y_pos = y_pos
        super().__init__()
    def get_position(self, image, objects):
        return self.get_absolute_position(image, objects[-1], self.x_pos, self.y_pos)