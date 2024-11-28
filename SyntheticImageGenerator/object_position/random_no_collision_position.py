from . import BasePositionDeterminer
from typing import List
import random
from ..utilities import register_positionDeterminer, logger

from . import BasePositionDeterminer
import random
from typing import List, Tuple

@register_positionDeterminer
class RandomPositionNoCollisionDeterminer(BasePositionDeterminer):
    def __init__(self, bounds: List[float] = [0, 1, 0, 1], max_attempts: int = 100):
        super().__init__()
        self.bounds = bounds
        self.max_attempts = max_attempts

    def get_position(self, image, objects):
        height, width = image.shape[:2]
        attempts = 0
        new_object = objects[-1]
        if len(objects) == 1:
            return self.get_absolute_position(image, new_object, random.uniform(self.bounds[0], self.bounds[1]), random.uniform(self.bounds[2], self.bounds[3]))
        while attempts < self.max_attempts:
            x_position = random.uniform(self.bounds[0], self.bounds[1])
            y_position = random.uniform(self.bounds[2], self.bounds[3])
            
            x_min, y_min, x_max, y_max = self.get_absolute_position(image, new_object, x_position, y_position)
            

            if not self._collides_with_existing_objects(x_min, y_min, x_max, y_max, objects):
                logger.info(f"Found position at ({x_position}, {y_position}) after {attempts} attempts.")
                return x_min, y_min, x_max, y_max

            attempts += 1
        logger.warning(f"Failed to find a collision-free position after {self.max_attempts} attempts. Returning random position.")
        return self.get_absolute_position(image, new_object, random.uniform(self.bounds[0], self.bounds[1]), random.uniform(self.bounds[2], self.bounds[3]))

    def _collides_with_existing_objects(self, x_min, y_min, x_max, y_max, objects):
        for obj in objects[:-1]:
            if self._check_collision(x_min, y_min, x_max, y_max, obj):
                return True
        return False

    def _check_collision(self, x1, y1, x2, y2, obj):
        h1, w1 = x2-x1, y2-y1
        x2, y2, w2, h2 = obj.bbox.coordinates
        return not (x1 + w1 < x2 or x1 > x2 + w2 or y1 + h1 < y2 or y1 > y2 + h2)
