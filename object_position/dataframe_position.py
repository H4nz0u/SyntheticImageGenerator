from . import BasePositionDeterminer
from typing import List
import random
from utilities import register_positionDeterminer, get_cached_dataframe, logger
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
            filtered_data = self.data[self.data['class'] == cls]
            x_values = filtered_data[self.x_column].dropna()
            y_values = filtered_data[self.y_column].dropna()
            if len(x_values) == 0:
                raise ValueError(f"No valid x_position values found for class '{self.cls}' in column '{self.x_column}'.")
            if len(y_values) == 0:
                raise ValueError(f"No valid x_position values found for class '{self.cls}' in column '{self.y_column}'.")
            x_position = np.random.choice(x_values)
            y_position = np.random.choice(y_values)
            return self.get_absolute_position(image, objects[-1], x_position, y_position)
        except Exception as e:
            logger.error(f"Failed to select an position for class '{cls}': {e}")
            raise