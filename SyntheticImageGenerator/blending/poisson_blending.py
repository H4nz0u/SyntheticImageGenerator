from .base_blending import BaseBlending
import cv2
import numpy as np
from typing import Tuple
from ..utilities import register_blending

@register_blending
class PoissonBlending(BaseBlending):
    def __init__(self, kernel_size: int = 15, padding: int = 50):
        """
        Initialize the PoissonBlending with specified kernel size and padding for morphological operations.
        
        Parameters:
        - kernel_size: Size of the kernel used for erosion and dilation to create inner and outer masks.
        - padding: Number of pixels to pad around the mask to ensure the outer mask fully encompasses the object.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

    def blend(
            self,
            background: np.ndarray,
            foreground: np.ndarray,
            mask: np.ndarray,
            position: Tuple[int, int, int, int]
        ) -> np.ndarray:
        """
        Perform Poisson blending with a double mask approach, blending only the boundary.
        
        Parameters:
        - background: Destination image (background).
        - foreground: Source image (object to be blended).
        - mask: Binary mask of the object in the source image.
        - position: Tuple (xmin, ymin, xmax, ymax) representing the bounding box to place foreground in background.
        
        Returns:
        - result: Blended image.
        """
        # Unpack position tuple
        xmin, ymin, xmax, ymax = position

        # Calculate width and height
        w = xmax - xmin
        h = ymax - ymin

        # Validate width and height
        if w <= 0 or h <= 0:
            raise ValueError("Invalid position values: xmax must be greater than xmin and ymax must be greater than ymin.")

        # Load background dimensions
        bg_h, bg_w = background.shape[:2]

        # Resize foreground and mask to the desired size
        foreground_resized = cv2.resize(foreground, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Ensure mask is binary
        _, mask_binary = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)

        # Apply padding to the mask to ensure the outer mask fully encompasses the object
        mask_padded = cv2.copyMakeBorder(
            mask_binary,
            self.padding,
            self.padding,
            self.padding,
            self.padding,
            cv2.BORDER_CONSTANT,
            value=0
        )

        # Adjust the foreground image to accommodate padding
        foreground_padded = cv2.copyMakeBorder(
            foreground_resized,
            self.padding,
            self.padding,
            self.padding,
            self.padding,
            cv2.BORDER_CONSTANT,
            value=0
        )

        # Calculate new position to account for padding
        padded_x = xmin - self.padding
        padded_y = ymin - self.padding

        # Ensure the new position is within the background boundaries
        padded_x = max(padded_x, 0)
        padded_y = max(padded_y, 0)

        # Check if padded foreground exceeds background dimensions
        if (padded_x + w + 2 * self.padding > bg_w) or (padded_y + h + 2 * self.padding > bg_h):
            raise ValueError("Padded foreground exceeds background dimensions. Adjust padding or position.")

        # Create inner and outer masks using morphological operations on the padded mask
        inner_mask = cv2.erode(mask_padded, self.kernel, iterations=1)
        outer_mask = cv2.dilate(mask_padded, self.kernel, iterations=1)

        # Create boundary mask by subtracting inner_mask from outer_mask
        boundary_mask = cv2.subtract(outer_mask, inner_mask)

        # Ensure masks are binary and 8-bit
        inner_mask = np.uint8(inner_mask > 0) * 255
        outer_mask = np.uint8(outer_mask > 0) * 255
        boundary_mask = np.uint8(boundary_mask > 0) * 255

        # Debug: Save masks to verify
        cv2.imwrite("inner_mask_padded.png", inner_mask)
        cv2.imwrite("outer_mask_padded.png", outer_mask)
        cv2.imwrite("boundary_mask.png", boundary_mask)
        print(f"inner_mask_padded shape: {inner_mask.shape}")
        print(f"outer_mask_padded shape: {outer_mask.shape}")
        print(f"Padded Position: ({padded_x}, {padded_y})")

        # Calculate the center for seamlessClone based on the padded position
        center_x = padded_x + (w + 2 * self.padding) // 2
        center_y = padded_y + (h + 2 * self.padding) // 2
        center = (center_x, center_y)

        # Ensure center is within background boundaries
        center_x = min(max(center[0], (w + 2 * self.padding) // 2), bg_w - (w + 2 * self.padding) // 2)
        center_y = min(max(center[1], (h + 2 * self.padding) // 2), bg_h - (h + 2 * self.padding) // 2)
        center = (center_x, center_y)
        print(f"Final Center: {center}")

        # First, blend the boundary mask with the background using NORMAL_CLONE
        blended_outer = cv2.seamlessClone(foreground_padded, background, boundary_mask, center, cv2.NORMAL_CLONE)
        cv2.imwrite("blended_outer.png", blended_outer)

        # Now, overlay the inner area from the foreground onto the blended image
        # Define the ROI coordinates in the blended image
        roi_x1 = padded_x
        roi_y1 = padded_y
        roi_x2 = padded_x + w + 2 * self.padding
        roi_y2 = padded_y + h + 2 * self.padding

        # Extract the blended region of interest (ROI) from the blended image
        blended_roi = blended_outer[roi_y1:roi_y2, roi_x1:roi_x2]

        # Extract the inner_mask area from the padded inner mask
        inner_mask_cropped = inner_mask[self.padding:self.padding + h, self.padding:self.padding + w]

        # Extract the inner area from the foreground_resized
        foreground_cropped = foreground_resized

        # Create a mask for the inner area in the ROI coordinates
        # Initialize a blank mask with the size of blended_roi
        if len(blended_roi.shape) == 3:
            full_inner_mask = np.zeros((blended_roi.shape[0], blended_roi.shape[1]), dtype=np.uint8)
        else:
            full_inner_mask = np.zeros_like(blended_roi, dtype=np.uint8)

        # Place the inner_mask_cropped into the full_inner_mask
        full_inner_mask[self.padding:self.padding + h, self.padding:self.padding + w] = inner_mask_cropped

        # Ensure the inner mask is binary
        full_inner_mask = np.uint8(full_inner_mask > 0) * 255

        # Create a mask for the foreground to match the blended_roi size
        foreground_full = np.zeros_like(blended_roi, dtype=foreground_cropped.dtype)
        foreground_full[self.padding:self.padding + h, self.padding:self.padding + w] = foreground_cropped

        # Ensure both images have the same number of channels
        if len(foreground_full.shape) != len(blended_roi.shape):
            raise ValueError("Foreground and blended ROI must have the same number of channels.")

        # If the images have multiple channels, ensure the mask is applied to each channel
        if len(foreground_full.shape) == 3 and full_inner_mask.ndim == 2:
            full_inner_mask_3ch = cv2.merge([full_inner_mask] * 3)
        else:
            full_inner_mask_3ch = full_inner_mask

        # Mask the foreground_full with the full_inner_mask
        foreground_part = cv2.bitwise_and(foreground_full, full_inner_mask_3ch)

        # Create inverse mask for background
        if len(full_inner_mask.shape) == 2 and len(blended_roi.shape) == 3:
            inverse_inner_mask = cv2.bitwise_not(full_inner_mask)
            inverse_inner_mask_3ch = cv2.merge([inverse_inner_mask] * 3)
        else:
            inverse_inner_mask = cv2.bitwise_not(full_inner_mask)

        # Extract the background part where the inner mask is not present
        background_part = cv2.bitwise_and(blended_roi, blended_roi, mask=inverse_inner_mask)

        # Combine the background and foreground parts
        blended_final_roi = cv2.add(background_part, foreground_part)

        # Place the blended_final_roi back into the blended image
        blended_outer[roi_y1:roi_y2, roi_x1:roi_x2] = blended_final_roi

        # Save the final blended image
        cv2.imwrite("blended_final.png", blended_outer)

        return blended_outer
