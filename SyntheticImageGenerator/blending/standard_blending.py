from .base_blending import BaseBlending
from ..utilities import register_blending
import numpy as np
from typing import Tuple
import cv2

@register_blending
class StandardBlending(BaseBlending):
    def __init__(self):
        super().__init__()
    
    def blend(
            self,
            background: np.ndarray,
            foreground: np.ndarray,
            mask: np.ndarray,
            position: Tuple[int, int, int, int]
        ) -> np.ndarray:
        x_start, y_start, x_end, y_end = position
        if len(mask.shape) == 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            roi = background[y_start:y_end, x_start:x_end]
            background[y_start:y_end, x_start:x_end] = roi * (1 - mask) + foreground * mask      
        return background