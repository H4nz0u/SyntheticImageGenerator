from ..image_management import ImgObject
import numpy as np
import cv2
from ..utilities import logger
class Transformation:
    def apply(self, obj: ImgObject):
        raise NotImplementedError('The apply method must be implemented by the subclass')
    def __str__(self) -> str:
        return self.__class__.__name__
    def update_bbox_from_mask(self, mask):
        """
        Update the bounding box based on the mask.

        The bbox is in the format [xmin, ymin, xmax, ymax].

        Args:
            mask (np.ndarray): Binary mask where the object is marked with non-zero values.

        Returns:
            np.ndarray or None: Updated bbox in [xmin, ymin, xmax, ymax] format or None if no object is found.
        """
        # Convert to grayscale if mask has multiple channels
        if len(mask.shape) == 3 and mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Threshold to binary
        ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # Find coordinates of all non-zero pixels
        y_indices, x_indices = np.nonzero(thresh)
        if not y_indices.size or not x_indices.size:
            logger.warning("No non-zero pixels in the mask")
            return None  # or np.array([0, 0, 0, 0]) if you prefer to return an empty bbox
        # Calculate the bounding box from non-zero pixels
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()

        # The bounding box is [x_min, y_min, x_max, y_max]
        bbox = np.array([x_min, y_min, x_max, y_max])
        logger.info(f"New BBox coordinates: {bbox}")
        return bbox