import cv2
import numpy as np
import os
from typing import List
from pathlib import Path
from ..image_management import ImgObject
from ..object_position import BasePositionDeterminer
from ..annotations import BaseAnnotator
from ..utilities import logger
from ..filter.brightness import TargetBrightness
from ..filter import Filter
from ..blending import BaseBlending

class Scene:
    def __init__(self, background: np.ndarray) -> None:
        self.background = background
        self.foregrounds: List[ImgObject] = []
        self.filters: List[Filter] = []
        self.positionDeterminer: BasePositionDeterminer = None
        self.annotator: BaseAnnotator = None
        self.blender: BaseBlending = None

    def add_filter(self, filter: Filter):
        self.filters.append(filter)
        logger.info(f"Filter added: {filter.__class__.__name__}")

    def apply_filter(self):
        for filter in self.filters:
            if isinstance(filter, TargetBrightness):
                if not self.foregrounds:
                    logger.warning("No foregrounds available for TargetBrightness filter.")
                    continue
                self.background = filter.apply(self.background, self.foregrounds[0].mask)
                logger.info("Applied TargetBrightness filter.")
            else:
                self.background = filter.apply(self.background)
                logger.info(f"Applied filter: {filter.__class__.__name__}")

    def add_foreground(self, foreground: ImgObject):
        if self.positionDeterminer is None:
            logger.error("Position determiner not configured.")
            raise ValueError("Position determiner not configured.")
        if self.blender is None:
            logger.error("Blender not configured.")
            raise ValueError("Blender not configured.")
        if self.annotator is None:
            logger.error("Annotator not configured.")
            raise ValueError("Annotator not configured.")

        self.foregrounds.append(foreground)
        logger.info("Adding foreground.")
        logger.debug(f"Initial Background shape: {self.background.shape}")

        # Step 0: Crop the foreground image, mask, and segmentation to the object's bbox
        self._crop_foreground(foreground)

        # Step 1: Compute scaling factor and resize
        self._resize_foreground(foreground)

        # Step 2: Determine position on background
        x_start, y_start, x_end, y_end = self.positionDeterminer.get_position(self.background, self.foregrounds)
        logger.info(f"Determined position: x_start={x_start}, y_start={y_start}, x_end={x_end}, y_end={y_end}")

        # Step 3: Blend foreground into background
        self._blend_foreground(foreground, x_start, y_start, x_end, y_end)

        # Step 4: Update bbox coordinates to background position
        foreground.bbox.coordinates = np.array([x_start, y_start, x_end, y_end])
        logger.info(f"Final bbox set to: {foreground.bbox.coordinates}")

        logger.info(f"Final relative position: {x_start / self.background.shape[1]}, {y_start / self.background.shape[0]}")

        # Step 5: Adjust segmentation coordinates to background coordinates
        self._adjust_segmentation(foreground, x_start, y_start)

        # Step 6: Pad the mask to match the background shape
        self._pad_mask(foreground, x_start, y_start, x_end, y_end)
        # Step 7: Add annotations
        self.add_annotation(foreground)

    def _crop_foreground(self, foreground: ImgObject):
        if foreground.bbox.coordinates is not None and len(foreground.bbox.coordinates) == 4:
            x_min, y_min, x_max, y_max = foreground.bbox.coordinates.astype(int)
            # Ensure coordinates are within the image bounds
            x_min = max(0, min(x_min, foreground.image.shape[1] - 1))
            y_min = max(0, min(y_min, foreground.image.shape[0] - 1))
            x_max = max(x_min + 1, min(x_max, foreground.image.shape[1]))
            y_max = max(y_min + 1, min(y_max, foreground.image.shape[0]))

            # Crop image
            foreground.image = foreground.image[y_min:y_max, x_min:x_max]
            logger.debug(f"Cropped foreground image to bbox: {x_min, y_min, x_max, y_max}")

            # Crop mask
            if foreground.mask is not None and foreground.mask.size > 0:
                foreground.mask = foreground.mask[y_min:y_max, x_min:x_max]
                logger.debug("Cropped foreground mask to bbox.")

            # Adjust segmentation coordinates
            if foreground.segmentation is not None and foreground.segmentation.size > 0:
                foreground.segmentation = foreground.segmentation - np.array([x_min, y_min])
                logger.debug("Adjusted segmentation coordinates relative to cropped image.")

            # Reset bbox coordinates to (0,0) to (width,height)
            foreground.bbox.coordinates = np.array([0, 0, x_max - x_min, y_max - y_min])
            logger.info(f"Cropped foreground to bbox: {foreground.bbox.coordinates}")
        else:
            # If bbox is not set, assume object occupies the entire image
            foreground.bbox.coordinates = np.array([0, 0, foreground.image.shape[1], foreground.image.shape[0]])
            logger.warning("BBox not set. Assuming foreground occupies the entire image.")

    def _resize_foreground(self, foreground: ImgObject):
        fg_height, fg_width = foreground.image.shape[:2]
        bg_height, bg_width = self.background.shape[:2]

        scale_factor = min(1.0, bg_width / fg_width, bg_height / fg_height)
        new_width = int(fg_width * scale_factor)
        new_height = int(fg_height * scale_factor)

        if scale_factor < 1.0:
            # Resize image and mask
            foreground.image = cv2.resize(foreground.image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            logger.debug(f"Resized foreground image to: {new_width}x{new_height}")

            if foreground.mask is not None and foreground.mask.size > 0:
                foreground.mask = cv2.resize(foreground.mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                logger.debug("Resized foreground mask.")

            # Scale segmentation
            if foreground.segmentation is not None and foreground.segmentation.size > 0:
                foreground.segmentation = (foreground.segmentation * scale_factor).astype(int)
                logger.debug("Scaled segmentation coordinates.")

            # Update bbox based on resized image
            foreground.bbox.coordinates = np.array([0, 0, new_width, new_height])
            logger.debug(f"Updated bbox after resizing: {foreground.bbox.coordinates}")
        else:
            logger.debug("No resizing needed for foreground.")

    def _blend_foreground(self, foreground: ImgObject, x_start: int, y_start: int, x_end: int, y_end: int):
        if foreground.mask is not None and foreground.mask.size > 0:
            try:
                blended_background = self.blender.blend(
                    background=self.background,
                    foreground=foreground.image,
                    mask=foreground.mask,
                    position=(x_start, y_start, x_end, y_end)
                )
                self.background = blended_background
                logger.info("Blended foreground with mask.")
            except Exception as e:
                logger.error(f"Error blending with mask: {e}. Falling back to alpha blending.")
                self._alpha_blend_foreground(foreground, x_start, y_start, x_end, y_end)
        else:
            logger.info("No mask provided. Using alpha blending.")
            self._alpha_blend_foreground(foreground, x_start, y_start, x_end, y_end)

    def _alpha_blend_foreground(self, foreground: ImgObject, x_start: int, y_start: int, x_end: int, y_end: int):
        try:
            roi = self.background[y_start:y_end, x_start:x_end]
            blended_roi = cv2.addWeighted(roi, 1 - 0.5, foreground.image, 0.5, 0)
            self.background[y_start:y_end, x_start:x_end] = blended_roi.astype(np.uint8)
            logger.info("Alpha blended foreground without mask.")
        except Exception as e:
            logger.error(f"Error in alpha blending: {e}")

    def _adjust_segmentation(self, foreground: ImgObject, x_start: int, y_start: int):
        if foreground.segmentation is not None and foreground.segmentation.size > 0:
            foreground.segmentation = foreground.segmentation + np.array([x_start, y_start])
            logger.debug("Adjusted segmentation coordinates to background position.")
        else:
            logger.info("No segmentation to adjust.")

    def _pad_mask(self, foreground: ImgObject, x_start: int, y_start: int, x_end: int, y_end: int):
        if foreground.mask is not None:
            padded_mask = np.zeros(self.background.shape[:2], dtype=np.uint8)
            mask_height, mask_width = foreground.mask.shape[:2]
            try:
                padded_mask[y_start:y_start + mask_height, x_start:x_start + mask_width] = foreground.mask
                foreground.mask = padded_mask
                logger.debug("Padded mask to match background dimensions.")
            except Exception as e:
                logger.error(f"Error padding mask: {e}")
        else:
            logger.info("No mask to pad.")

    def configure_positioning(self, positionDeterminer: BasePositionDeterminer):
        self.positionDeterminer = positionDeterminer
        logger.info("Configured position determiner.")

    def configure_annotator(self, annotator: BaseAnnotator):
        self.annotator = annotator
        self.annotator.reset()
        logger.info("Configured annotator and reset annotations.")

    def configure_blending(self, blender: BaseBlending):
        self.blender = blender
        logger.info("Configured blending method.")

    def add_annotation(self, obj: ImgObject):
        if self.annotator is None:
            logger.error("Annotator not configured.")
            raise ValueError("Annotator not configured.")

        if obj.bbox.coordinates is not None and len(obj.bbox.coordinates) == 4:
            self.annotator.append_object(obj.bbox, obj.cls)
            logger.debug(f"Added bbox annotation for class: {obj.cls}")
        else:
            raise ValueError("Invalid or missing BBox and segmentation.")

    def write(self, path: Path, size: tuple, annotation: bool = True):
        try:
            image = cv2.resize(self.background, size)
            if annotation:
                if not path.parent.exists():
                    os.makedirs(path.parent)
                filename = path.name
                file_ending = filename.split(".")[-1]
                # Replace the file ending with xml
                xml_filename = f"{'.'.join(filename.split('.')[:-1])}.xml" if '.' in filename else f"{filename}.xml"
                xml_path = path.with_name(xml_filename)
                self.annotator.write_xml(xml_path, image.shape)
                logger.info(f"Annotations written to {xml_path}")
            cv2.imwrite(str(path), image)
            logger.info(f"Image written to {path}")
        except Exception as e:
            logger.error(f"Error writing image and annotations: {e}")

    def show(self, show_bbox: bool = True, show_mask: bool = True, 
             show_segmentation: bool = True, show_class: bool = True, output_path: str = "test_visualization.jpg"):
        display_image = self.background.copy()

        if show_bbox:
            self.show_bbox(display_image)
        if show_mask:
            self.show_mask(display_image)
        if show_segmentation:
            self.show_segmentation(display_image)
        if show_class:
            self.show_class(display_image)

        try:
            cv2.imwrite(output_path, display_image)
            logger.info(f"Visualization saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving visualization: {e}")

    def show_bbox(self, display_image: np.ndarray):
        for fg in self.foregrounds:
            if fg.bbox.coordinates is None or len(fg.bbox.coordinates) != 4:
                logger.warning("Invalid or missing BBox. Skipping bbox visualization.")
                continue

            x_min, y_min, x_max, y_max = fg.bbox.coordinates.astype(int)

            # Ensure coordinates are within image bounds
            x_min = np.clip(x_min, 0, display_image.shape[1] - 1)
            y_min = np.clip(y_min, 0, display_image.shape[0] - 1)
            x_max = np.clip(x_max, x_min + 1, display_image.shape[1])
            y_max = np.clip(y_max, y_min + 1, display_image.shape[0])
            # Draw bounding box
            cv2.rectangle(display_image, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
            logger.debug(f"Drew bbox: ({x_min}, {y_min}), ({x_max}, {y_max})")

    def show_mask(self, display_image: np.ndarray):
        for fg in self.foregrounds:
            if fg.bbox.coordinates is None or len(fg.bbox.coordinates) != 4:
                logger.warning("Invalid or missing BBox. Skipping mask visualization.")
                continue

            x_min, y_min, x_max, y_max = fg.bbox.coordinates.astype(int)
            w = x_max - x_min
            h = y_max - y_min

            # Ensure coordinates are within image bounds
            x_min = np.clip(x_min, 0, display_image.shape[1] - 1)
            y_min = np.clip(y_min, 0, display_image.shape[0] - 1)
            x_max = np.clip(x_max, x_min + 1, display_image.shape[1])
            y_max = np.clip(y_max, y_min + 1, display_image.shape[0])
            w = x_max - x_min
            h = y_max - y_min

            if fg.mask is not None:
                try:
                    # Resize the mask to match the bbox dimensions
                    resized_mask = cv2.resize(fg.mask, (w, h), interpolation=cv2.INTER_NEAREST)

                    # Ensure mask is binary
                    _, binary_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)

                    # Apply color map to the mask
                    colored_mask = cv2.applyColorMap(binary_mask, cv2.COLORMAP_JET)

                    # Blend the colored mask with the ROI
                    roi = display_image[y_min:y_max, x_min:x_max]
                    blended_roi = cv2.addWeighted(roi, 0.5, colored_mask, 0.5, 0)
                    display_image[y_min:y_max, x_min:x_max] = blended_roi
                    logger.debug(f"Applied mask to region: ({x_min}, {y_min}), ({x_max}, {y_max})")
                except Exception as e:
                    logger.error(f"Error in show_mask: {e}")

    def show_segmentation(self, display_image: np.ndarray):
        height, width = display_image.shape[:2]
        for fg in self.foregrounds:
            if hasattr(fg, 'segmentation') and fg.segmentation.size > 0:
                segmentation = fg.segmentation.copy()

                # Ensure segmentation is a 2D array with shape (N, 2)
                if segmentation.ndim != 2 or segmentation.shape[1] != 2:
                    logger.warning("Segmentation has an unexpected shape. Skipping.")
                    continue

                # Clip segmentation points to lie within image boundaries
                segmentation[:, 0] = np.clip(segmentation[:, 0], 0, width - 1)
                segmentation[:, 1] = np.clip(segmentation[:, 1], 0, height - 1)

                # Convert to integer type
                segmentation = segmentation.astype(int)

                # Ensure segmentation is in the correct shape for cv2.drawContours
                segmentation = segmentation.reshape((-1, 1, 2))

                # Close the contour if necessary
                if not np.array_equal(segmentation[0], segmentation[-1]):
                    segmentation = np.concatenate([segmentation, segmentation[0:1]], axis=0)

                # Draw the contour
                cv2.drawContours(display_image, [segmentation], -1, (0, 255, 0), thickness=cv2.FILLED)
                logger.debug("Drew segmentation contour.")
            else:
                logger.info("No segmentation to display for this foreground.")

    def show_class(self, display_image: np.ndarray):
        for fg in self.foregrounds:
            if fg.bbox.coordinates is None or len(fg.bbox.coordinates) != 4:
                logger.warning("Invalid or missing BBox. Skipping class label visualization.")
                continue

            x_min, y_min, x_max, y_max = fg.bbox.coordinates.astype(int)

            # Ensure coordinates are within image bounds
            x_min = np.clip(x_min, 0, display_image.shape[1] - 1)
            y_min = np.clip(y_min, 0, display_image.shape[0] - 1)

            if hasattr(fg, 'cls') and fg.cls:
                try:
                    cv2.putText(
                        display_image, 
                        fg.cls, 
                        (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (255, 255, 255), 
                        2, 
                        cv2.LINE_AA
                    )
                    logger.debug(f"Drew class label '{fg.cls}' at ({x_min}, {y_min - 10})")
                except Exception as e:
                    logger.error(f"Error drawing class label: {e}")
            else:
                logger.info("No class label to display for this foreground.")
