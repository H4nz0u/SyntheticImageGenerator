from typing import Tuple
class BasePositionDeterminer:
    def __init__(self):
        pass
    def get_position(self, image, objects) -> Tuple[float, float]:
        raise NotImplementedError('The get_position method must be implemented by the subclass')