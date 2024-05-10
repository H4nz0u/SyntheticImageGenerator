import cv2
import numpy as np
from noise import snoise2
import random
from utils import get_random_from_bounds


class SignImage:
    image: np.ndarray
    bounding_box: np.ndarray
    mask: np.ndarray
    random_gen: random.Random

    def __init__(
        self, image, bounding_box, random_gen: random.Random, **kwargs
    ) -> None:
        self.image = image
        self.bounding_box = bounding_box
        self.random_gen = random_gen
        self.scale = get_random_from_bounds(*kwargs.get("size"), self.random_gen)
        self.angle = get_random_from_bounds(*kwargs.get("rotation"), self.random_gen)
        self.shear_factor = get_random_from_bounds(*kwargs.get("shear_factor"), self.random_gen)
        self.tilt_angle = get_random_from_bounds(
            *kwargs.get("tilt_angle"), self.random_gen
        )

        self.fixed_rotation = kwargs.get("fixed_rotation")
        self.shadow = kwargs.get("shadow")
        self.shadow_frequency = kwargs.get("shadow_frequency")
        self.shadow_intensity = kwargs.get("shadow_intensity")
        self.enable_rotation = kwargs.get("enable_rotation")

    def _cut_out_sign(self):
        """
        Extracts and rectifies the sign from the image using the bounding box.

        Args:
            sign: The image of the sign.
            bounding_box: The bounding box.

        Returns:
            The rectified (straightened) cutout of the sign.
        """
        black_mask = np.all(self.image == [0, 0, 0], axis=-1)

        # Replace black pixels with the new color
        self.image[black_mask] = (0, 255, 0)
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [self.bounding_box], -1, 255, thickness=cv2.FILLED)
        black_mask = np.all(self.image == [0, 255, 0], axis=-1)

        # Replace black pixels with the new color
        self.image[black_mask] = (0, 0, 0)

        # Apply the mask to the image
        self.image = cv2.bitwise_and(self.image, self.image, mask=mask)
        self.mask = mask

    def _rectify_sign(self):
        """
        Straightens the sign image according to the detected rectangle.

        Args:
            sign: The image of the sign.
            bounding_box: The detected bounding box.

        Returns:
            The rectified (straightened) image of the sign.
        """
        # Compute minimal area rectangle
        rect = cv2.minAreaRect(self.bounding_box)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Get the width and height of the detected rectangle
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")

        # Coordinate of the points in box points after the rectangle has been
        # straightened
        dst_pts = np.array(
            [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
            dtype="float32",
        )

        # The perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # Directly warp the rotated rectangle to get the straightened rectangle
        self.image = cv2.warpPerspective(self.image, M, (width, height))
        self.bounding_box = cv2.perspectiveTransform(
            self.bounding_box.astype(float), M
        ).astype(int)
        if height < width:
            self._rotate(90)

    def _tilt_sign(self, tilt_angle):
        """
        Tilts the sign image by the specified angle using a perspective transformation.

        Args:
            tilt_angle: The angle (in degrees) to tilt the sign image.

        Returns:
            None (modifies self.image and self.bounding_box in-place)
        """

        # Convert the tilt angle from degrees to radians
        tilt_radians = -tilt_angle * (np.pi / 180.0)

        # Define transformation matrices
        proj2dto3d = np.array(
            [
                [1, 0, -self.image.shape[1] / 2],
                [0, 1, -self.image.shape[0] / 2],
                [0, 0, 0],
                [0, 0, 1],
            ],
            np.float32,
        )

        ry = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )

        trans = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 230],  # 400 to move the image in z axis
                [0, 0, 0, 1],
            ],
            np.float32,
        )

        proj3dto2d = np.array(
            [
                [200, 0, self.image.shape[1] / 2, 0],
                [0, 200, self.image.shape[0] / 2, 0],
                [0, 0, 1, 0],
            ],
            np.float32,
        )

        # Define y rotation matrix
        ry[0, 0] = np.cos(tilt_radians)
        ry[0, 2] = -np.sin(tilt_radians)
        ry[2, 0] = np.sin(tilt_radians)
        ry[2, 2] = np.cos(tilt_radians)

        output_size = (int(self.image.shape[1] * 1.5), int(self.image.shape[0] * 1.5))
        
        final = proj3dto2d @ (trans @ (ry @ proj2dto3d))
        # Apply the perspective transformation to the sign image
        self.image = cv2.warpPerspective(
            self.image,
            final,
            output_size,
            None,
            cv2.INTER_LINEAR,
            cv2.BORDER_CONSTANT,
            (0, 0, 0),
        )
        # Transform the bounding box points
        for i in range(self.bounding_box.shape[0]):
            point = self.bounding_box[i]
            # Add a third dimension to the bounding box coordinates
            point_3d = np.append(point, [1])
            # Homogeneous coordinates transformation
            point_transformed = final @ point_3d
            # Return to 2D coordinates and save the result
            self.bounding_box[i] = (point_transformed / point_transformed[2])[:2]

    def _scale_down_image(self):
        """
        Scales down an image.

        Args:
            image: The image to scale.
            scale_factor: The factor by which to scale down the image.

        Returns:
            The scaled down image.
        """
        new_width = int(self.image.shape[1] * self.scale)
        new_height = int(self.image.shape[0] * self.scale)
        new_dimensions = (new_width, new_height)
        resized_image = cv2.resize(
            self.image, new_dimensions, interpolation=cv2.INTER_AREA
        )
        self.image = resized_image
        # Update bounding box
        self.bounding_box = (self.bounding_box * self.scale).astype(int)
    
    def _rotate_random(self, offset: int):
        angle = self.random_gen.choice([90, 180, 270, 0]) + offset
        self._rotate(angle)

    def _rotate(self, angle):
        # Image center
        image_center = tuple(np.array(self.image.shape[1::-1]) / 2)
        
        # Original dimensions
        height, width = self.image.shape[:2]
        
        # Calculate the rotation matrix around the image center
        M = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        
        # Calculate the new dimensions required to avoid clipping the rotated image
        # This uses the absolute cos and sin values to ensure positive dimensions
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        
        # New dimensions to fit the rotated image
        new_w = int((height * sin) + (width * cos))
        new_h = int((height * cos) + (width * sin))
        
        # Adjusting the rotation matrix to account for translation
        M[0, 2] += (new_w / 2) - image_center[0]
        M[1, 2] += (new_h / 2) - image_center[1]
        
        # Apply the rotation
        self.image = cv2.warpAffine(self.image, M, (new_w, new_h))

        # Update bounding box
        self.bounding_box = cv2.transform(self.bounding_box.astype(float), M).astype(int)

    def process(self):
        # Cut out and resize the sign
        if self.shadow:
            self._generate_shadow()
        # self._rectify_sign()
        self._scale_down_image()
        if self.fixed_rotation != 0:
            self._rotate(self.fixed_rotation)
        if self.enable_rotation:
            self._rotate_random(self.angle)
        if self.tilt_angle != 0:
            self._tilt_sign(self.tilt_angle)
        self._shear()
        self._cut_out_sign()
        #self._compress_dynamic_range(20, 200)

    def _generate_shadow(self):
        # Create an empty image to store the Perlin noise
        rows, cols, _ = self.image.shape
        shadow_img = np.zeros((rows, cols), dtype=np.float32)

        # Fill the image with Perlin noise
        for i in range(rows):
            for j in range(cols):
                shadow_img[i][j] = (
                    snoise2(
                        i / (rows / self.shadow_frequency),
                        j / (cols / self.shadow_frequency),
                    )
                    + 1
                ) / 2  # Adjust the divisor for noise frequency

        # Create a shadow mask
        shadow_mask = np.zeros_like(self.image, dtype=np.float32)
        shadow_mask.fill(
            self.shadow_intensity
        )  # Fill with the shadow intensity, adjust this value as per your requirements

        # Modulate the shadow mask with the Perlin noise
        shadow_mask *= np.expand_dims(shadow_img, axis=2)

        # Blend the original image with the shadow mask
        img_float = self.image.astype(np.float32) / 255
        blended = cv2.addWeighted(img_float, 0.7, shadow_mask, -0.5, 0.3)
        blended = np.clip(blended, 0.01, 1)  # Ensure pixel values fall within [0, 1]

        self.image = (blended * 255).astype(np.uint8)

    
    def _shear(self):
        height, width = self.image.shape[:2]
        M = np.array([[1, self.shear_factor, 0], [0, 1, 0]], dtype=float)
        self.image = cv2.warpAffine(self.image, M, (int(width + height * self.shear_factor), height))
        self.bounding_box = cv2.transform(self.bounding_box.astype(float), M).astype(int)

    def _compress_dynamic_range(self, new_min, new_max):
        # Original min and max
        original_min, original_max = self.image.min(), self.image.max()
        
        # Scale and shift the pixel values
        compressed_image = (self.image - original_min) / (original_max - original_min) * (new_max - new_min) + new_min
        compressed_image = np.clip(compressed_image, new_min, new_max)  # Ensure values are within the new range
        
        return compressed_image.astype(np.uint8)