from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

class BaseBlending(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def blend(
            self,
            background: np.ndarray,
            foreground: np.ndarray,
            mask: np.ndarray,
            position: Tuple[int, int, int, int]
        ) -> np.ndarray:
        pass