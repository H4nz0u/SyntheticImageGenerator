from .base_transform import BaseTransform
from image_management.image import Image
import cv2
class Rotate(BaseTransform):
    def __init__(self, angle):
        self.angle = angle
    def apply(self, image: cv2.typing.MatLike):
        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)