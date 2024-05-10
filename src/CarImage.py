import math
import numpy as np
import cv2
from typing import Tuple
from SignImage import SignImage
import os
import random
from utils import get_random_from_bounds


class CarImage:
    image: np.ndarray
    signImage: SignImage
    random_gen: random.Random

    def __init__(
        self, image: np.array, signImage: np.array, random_gen: random.Random, **kwargs
    ) -> None:
        self.random_gen = random_gen
        self.image = image
        self.signImage = signImage

        self.position = [
            get_random_from_bounds(
                -kwargs.get("position"), kwargs.get("position"), self.random_gen
            )
            for _ in range(2)
        ]

        self.noise = kwargs.get("noise")
        self.noise_intensity = kwargs.get("noise_intensity")
        self.noise_correlation = kwargs.get("noise_correlation")
        self.monochrome = kwargs.get("monochrome")
        self.zoom = kwargs.get("zoom")
        self.zoom_factor = kwargs.get("zoom_factor")
        self.zoom_position = kwargs.get("zoom_position")
        self.black_and_white = kwargs.get("make_black_white")

    def _add_random_noise(self):
        """
        Adds random noise to an image.

        Args:
            image: The image to add noise to.
            intensity: The intensity of the noise.

        Returns:
            The image with added noise.
        """
        # Create a low-resolution noise image
        low_res_size = (
            self.image.shape[1] // 300,
            self.image.shape[0] // 300,
            self.image.shape[2],
        )  # adjust divisor for level of coherence
        low_res_noise = np.random.normal(scale=self.noise_intensity, size=low_res_size)

        # Upscale to original size
        noise = cv2.resize(
            low_res_noise,
            (self.image.shape[1], self.image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        # Clip values to ensure they fall within the valid range
        noise = np.clip(noise, -1.0, 1.0)

        # Convert image to float
        img_float = self.image.astype(np.float32) / 255.0

        # Apply noise to the image
        img_noisy = img_float + noise

        # Clip values to ensure they fall within the valid range
        img_noisy = np.clip(img_noisy, 0.0, 1.0)

        self.image = (img_noisy * 255).astype(np.uint8)

    def _place_sign_on_car(self):
        """
        Places a rotated sign image onto a car image.

        Args:
            car_image: The image of the car.
            sign_image: The image of the sign.
            bounding_box: The bounding box of the sign.
            position: The relative position to place the sign on the car.

        Returns:
            The image of the car with the sign placed on it.
        """
        sign_h, sign_w = self.signImage.image.shape[:2]
        car_h, car_w = self.image.shape[:2]

        car_center_x, car_center_y = (
            car_w // 2 + int(car_w * self.position[0]),
            car_h // 2 + int(car_h * self.position[1]),
        )
        sign_center_x, sign_center_y = sign_w // 2, sign_h // 2

        # Calculate the position of the sign on the car image
        x_start = car_center_x - sign_center_x
        x_end = x_start + sign_w
        y_start = car_center_y - sign_center_y
        y_end = y_start + sign_h

        # Ensure the entire sign is within the bounds of the car image
        x_start = max(min(x_start, car_w - sign_w), 0)
        y_start = max(min(y_start, car_h - sign_h), 0)

        x_end = min(x_start + sign_w, car_w)
        y_end = min(y_start + sign_h, car_h)

        # Shift the bounding box points to their position on the car image
        self.signImage.bounding_box += [x_start, y_start]

        mask = self.signImage.mask
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask = mask / 255.0

        # Place the sign on the car image using the feathered mask
        car_image_with_sign = self.image.copy()
        for c in range(3):  # For each color channel
            car_channel = self.image[y_start:y_end, x_start:x_end, c]
            sign_channel = self.signImage.image[
                : (y_end - y_start), : (x_end - x_start), c
            ]
            mask_channel = mask[: (y_end - y_start), : (x_end - x_start)]

            blended_channel = (
                car_channel * (1 - mask_channel) + sign_channel * mask_channel
            )
            car_image_with_sign[y_start:y_end, x_start:x_end, c] = blended_channel

        # Draw bounding box on the final image for testing
        """self.image = cv2.polylines(
            car_image_with_sign, [self.signImage.bounding_box], True, (0, 255, 0), 2
        )"""
        self.image = car_image_with_sign

    def calculate_zoom_parameters(self):
        img_height, img_width, _ = self.image.shape

        # Calculate the center of the image
        Cx, Cy = img_width / 2, img_height / 2

        # Calculate the upper and lower bounds for x and y
        Ux, Uy = Cx + (Cx * self.zoom_position[0]), Cy + (Cy * self.zoom_position[0])
        Lx, Ly = Cx - (Cx * self.zoom_position[1]), Cy - (Cy * self.zoom_position[1])

        # Randomly pick the coordinates to zoom in on
        Px, Py = self.random_gen.uniform(Lx, Ux), self.random_gen.uniform(Ly, Uy)

        # Randomly pick the zoom factor
        Z = self.random_gen.uniform(self.zoom_factor[0], self.zoom_factor[1])

        # Calculate new dimensions based on zoom factor
        new_width, new_height = img_width / Z, img_height / Z

        # Calculate the coordinates of the zoom window
        x1, y1 = Px - new_width / 2, Py - new_height / 2
        x2, y2 = Px + new_width / 2, Py + new_height / 2

        # Make sure the zoom window lies within the image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_width, x2), min(img_height, y2)

        return Z, (int(x1), int(x2), int(y1), int(y2))
    
    def match_histograms(self):
        car_image_lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2Lab)
        sign_image_lab = cv2.cvtColor(self.signImage.image, cv2.COLOR_BGR2Lab)
        result_color_chanels = []
        for template, source in zip(cv2.split(car_image_lab), cv2.split(sign_image_lab)):
            source_values, indices, counts = np.unique(source.ravel(), return_inverse=True, return_counts=True)
            template_values = np.unique(template.ravel())
            
            s_cdf = np.cumsum(counts).astype(np.float64)
            s_cdf /= s_cdf[-1]
            t_cdf = np.cumsum(np.histogram(template, bins=template_values.size, range=(0,256))[0]).astype(np.float64)
            t_cdf /= t_cdf[-1]
            
            interpolated = np.interp(s_cdf, t_cdf, template_values).astype(np.uint8)
            result_color_chanels.append(interpolated[indices].reshape(source.shape))
        print(len(result_color_chanels))
        self.signImage.image = cv2.cvtColor(cv2.merge(result_color_chanels), cv2.COLOR_Lab2BGR)
        

    def apply_zoom(self, zoom_window_coords):
        """
        Apply zoom transformation to the image.

        Args:
        zoom_factor (float): The zoom factor.
        center_point (tuple): The center point for zooming (x, y).

        Returns:
        numpy array: The zoomed image.
        """
        x1, x2, y1, y2 = zoom_window_coords

        self.image = self.image[y1:y2, x1:x2]

    def make_monochrome(self):
        tainted_img = np.zeros_like(self.image)

        tainted_img[:, :] = [
            self.random_gen.randint(0, 255),
            self.random_gen.randint(0, 255),
            self.random_gen.randint(0, 255),
        ]  # Yellow color in BGR format

        alpha = 0.85  # Adjust this value to control the intensity of the yellow tint
        self.image = cv2.addWeighted(self.image, alpha, tainted_img, 1 - alpha, 0)
        
    def _apply_noise(self):
        denoised_car = cv2.fastNlMeansDenoisingColored(self.image, None, 10, 10, 7, 21)
        noise = cv2.absdiff(self.image, denoised_car)
        noise = np.uint8(noise)
        # Get dimensions of the second image
        height2, width2 = self.signImage.image.shape[:2]

        # Resize the noise to match the dimensions of the second image
        resized_noise = cv2.resize(noise, (width2, height2), interpolation=cv2.INTER_LINEAR)

        # Add the resized noise to the second image
        # Note: cv2.add ensures that values are clipped between 0 and 255
        sign_with_noise = cv2.add(self.signImage.image, resized_noise)

        # Convert to uint8 if not already
        self.signImage.image = np.uint8(sign_with_noise)

    def process(self):
        original_shape = (self.image.shape[1], self.image.shape[0])

        # First, zoom in on a random region of the background
        if self.zoom:
            zoom_factor, zoom_window = self.calculate_zoom_parameters()
            self.apply_zoom(zoom_window)

            # Scale the background back to the original resolution
            self.image = cv2.resize(
                self.image, original_shape, interpolation=cv2.INTER_LINEAR
            )
        #self.match_histograms()
        #self._apply_noise()
        # Next, place the sign onto the car image
        self._place_sign_on_car()
        # If noise is enabled, add random noise to the image
        if self.noise:
            self._add_random_noise()

        # If monochrome is enabled, apply the monochrome effect to the image
        if self.monochrome:
            self.make_monochrome()
        
        if self.black_and_white:
            self._make_black_white()

    def _make_black_white(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        alpha = 0.3
        self.image = cv2.equalizeHist(self.image)
        self.image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=0)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

    def write(self, path, filename):
        # Save image
        if not os.path.isdir(path):
            os.makedirs(path)
        cv2.imwrite(os.path.join(path, filename), self.image)
