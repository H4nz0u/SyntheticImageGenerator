from . import BasePositionDeterminer
from typing import List
import random
from ..utilities import register_positionDeterminer, get_cached_dataframe, logger
import numpy as np
@register_positionDeterminer
class DataframePositionDeterminer(BasePositionDeterminer):
    def __init__(self, dataframe_path, column_names: List[str] = ['upper_left_x', 'upper_left_y']):
        super().__init__()
        self.dataframe_path = dataframe_path
        self.x_column, self.y_column = column_names
        self.data = get_cached_dataframe(self.dataframe_path)
    def get_position(self, image, objects):
        try:
            cls = objects[-1].cls
            x_position = self.data.sample_parameter(self.x_column, {'class': cls})
            y_position = self.data.sample_parameter(self.y_column, {'class': cls})
            return self.get_absolute_position(image, objects[-1], x_position, y_position)
        except Exception as e:
            logger.error(f"Failed to select an position for class '{objects[-1].cls}': {e}")
            raise