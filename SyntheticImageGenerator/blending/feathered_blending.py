from typing import Tuple
import numpy as np
import cv2
from .base_blending import BaseBlending
from ..utilities import register_blending

@register_blending
class FeatheredBlending(BaseBlending):
    def __init__(self, feather_amount: int = 25):
        """
        Initialize the FeatheredBlending class.

        :param feather_amount: Kernel size for Gaussian blur to feather the mask edges.
        """
        super().__init__()
        self.feather_amount = feather_amount

    def blend(
        self,
        background: np.ndarray,
        foreground: np.ndarray,
        mask: np.ndarray,
        position: Tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax)
    ) -> np.ndarray:
        """
        Blend the foreground onto the background with feathered edges.

        :param background: Background image as a NumPy array (H, W, 3).
        :param foreground: Foreground image as a NumPy array (H, W, 3).
        :param mask: Binary mask as a NumPy array (H, W).
        :param position: Tuple (xmin, ymin, xmax, ymax) specifying placement on background.
        :return: Blended image as a NumPy array.
        """
        x_min, y_min, x_max, y_max = position
        width = x_max - x_min
        height = y_max - y_min

        bg_h, bg_w = background.shape[:2]

        # Boundary Checks
        if x_min < 0 or y_min < 0:
            raise ValueError(f"Position coordinates cannot be negative: ({x_min}, {y_min}, {x_max}, {y_max})")
        if x_max > bg_w or y_max > bg_h:
            raise ValueError(f"Foreground exceeds background dimensions: ({x_min}, {y_min}, {x_max}, {y_max}) vs Background ({bg_w}, {bg_h})")

        # Feather the mask edges
        feathered_mask = cv2.GaussianBlur(mask, (self.feather_amount, self.feather_amount), 0)
        feathered_mask = feathered_mask.astype(float) / 255.0  # Normalize to [0,1]

        # Ensure mask has three channels
        if len(feathered_mask.shape) == 2:
            feathered_mask = cv2.merge([feathered_mask, feathered_mask, feathered_mask])

        # Define the region of interest (ROI) on the background
        roi = background[y_min:y_max, x_min:x_max]

        # Convert images to float for blending
        foreground_float = foreground.astype(float) / 255.0
        roi_float = roi.astype(float) / 255.0
        
        # Perform alpha blending
        blended_roi = feathered_mask * foreground_float + (1 - feathered_mask) * roi_float

        # Convert blended ROI back to uint8
        blended_roi_uint8 = np.clip(blended_roi * 255, 0, 255).astype(np.uint8)

        # Replace the ROI on the background with the blended ROI
        blended_image = background.copy()
        blended_image[y_min:y_max, x_min:x_max] = blended_roi_uint8

        return blended_image
