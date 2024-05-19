from . import BasePositionDeterminer
from typing import List
import random
from utilities import register_positionDeterminer

@register_positionDeterminer
class RandomPositionDeterminer(BasePositionDeterminer):
    def __init__(self, bounds: List[float] = [0, 1, 0, 1]):
        super().__init__()
        self.bounds = bounds
    def get_position(self, image, objects):
        x_position = random.uniform(self.bounds[0], self.bounds[1])
        y_position = random.uniform(self.bounds[2], self.bounds[3])
        return x_position, y_position 