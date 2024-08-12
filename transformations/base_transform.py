from image_management import ImgObject
import numpy as np
import cv2
class Transformation:
    def apply(self, image: ImgObject):
        raise NotImplementedError('The apply method must be implemented by the subclass')
    def __str__(self) -> str:
        return self.__class__.__name__
    def update_bbox_from_mask(self, mask):
        # Assuming mask is a binary image where the object is marked with non-zero (True) values
        if len(mask.shape) == 3 and mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # Find coordinates of all non-zero pixels
        y_indices, x_indices = np.nonzero(thresh)
        if not y_indices.size or not x_indices.size:
            return None  # or (0, 0, 0, 0) if you prefer to return an empty bbox

        # Calculate the bounding box from non-zero pixels
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()

        # The bounding box is (x_min, y_min, width, height)
        bbox = np.array([x_min, y_min, x_max - x_min, y_max - y_min])
        return bbox