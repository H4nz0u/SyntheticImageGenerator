from . import BasePositionDeterminer
from utilities.registry import register_positionDeterminer

@register_positionDeterminer
class CenterPositionDeterminer(BasePositionDeterminer):
    def get_position(self, image, objects):
        return self.get_absolute_position(image, objects[-1], 0.5, 0.5)