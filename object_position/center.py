from . import BasePositionDeterminer
from utilities.registry import register_positionDeterminer

@register_positionDeterminer
class CenterPositionDeterminer(BasePositionDeterminer):
    def get_position(self):
        return 0.5, 0.5