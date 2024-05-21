import cv2
from image_management import ImgObject
import numpy as np
from object_position import BasePositionDeterminer
from annotations import BaseAnnotator
import os
class Scene:
    def __init__(self, background) -> None:
        self.background = background
        self.foregrounds = []
        self.filters = []
    
    def add_filter(self, filter):
        self.filters.append(filter)
    
    def apply_filter(self):
        for filter in self.filters:
            self.background = filter.apply(self.background)
            
    def add_foreground(self, foreground: ImgObject):
        position_x, position_y = self.positionDeterminer.get_position(self.background, self.foregrounds)
        
        self.foregrounds.append(foreground)
        obj_h, obj_w = foreground.image.shape[:2]  # Use original dimensions
        background_h, background_w = self.background.shape[:2]

        # Calculate insertion point based on determined position
        x_start = int((background_w - obj_w) * position_x)
        y_start = int((background_h - obj_h) * position_y)

        # Ensure the insertion point keeps the entire object within the background
        x_start = max(min(x_start, background_w - obj_w), 0)
        y_start = max(min(y_start, background_h - obj_h), 0)

        # Calculate end points, ensuring they do not exceed the background
        x_end = min(x_start + obj_w, background_w)
        y_end = min(y_start + obj_h, background_h)

        # Clipping dimensions if necessary
        clipped_width = x_end - x_start
        clipped_height = y_end - y_start

        # Use clipped image regions
        clipped_image = foreground.image[:clipped_height, :clipped_width]
        clipped_mask = foreground.mask[:clipped_height, :clipped_width]

        # Normalize and prepare mask for blending
        mask = clipped_mask.astype(np.float32) / 255.0
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if len(mask.shape) == 2 else mask

        # Retrieve ROI from the background
        roi = self.background[y_start:y_end, x_start:x_end]

        # Blend the clipped foreground onto the background
        self.background[y_start:y_end, x_start:x_end] = roi * (1 - mask) + clipped_image * mask

        foreground.bbox.coordinates += np.array([x_start, y_start, 0, 0])
        foreground.segmentation += np.array([x_start, y_start])
        foreground.mask = mask


    def configure_positioning(self, positionDeterminer: BasePositionDeterminer):
        self.positionDeterminer = positionDeterminer

    def configure_annotator(self, annotator: BaseAnnotator):
        self.annotator = annotator

    def write(self, path, size, annotation=True):
        image = cv2.resize(self.background, size)
        if annotation:
            filename = os.path.basename(path)
            file_ending = filename.split(".")[-1]
            xml_path = path.replace(file_ending, "xml")
            for obj in self.foregrounds:
                self.annotator.append_object(obj.segmentation, obj.cls)
            self.annotator.write_xml(xml_path, image.shape)
        cv2.imwrite(path, image)

    def show(self, show_bbox=True, show_mask=True, show_segmentation=True, show_class=True):
        display_image = self.background.copy()
        
        if show_bbox:
            self.show_bbox(display_image)
        if show_mask:
            self.show_mask(display_image)
        if show_segmentation:
            self.show_segmentation(display_image)
        if show_class:
            self.show_class(display_image)

        cv2.imshow("Scene with Annotations", cv2.resize(display_image, (800, 600)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_bbox(self, display_image):
        for fg in self.foregrounds:
            x, y, w, h = fg.bbox.coordinates.astype(int)
            x = min(max(x, 0), display_image.shape[1] - 1)
            y = max(min(y, display_image.shape[0] - 1), 0)
            # Draw bounding box
            cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    def show_mask(self, display_image):
        for fg in self.foregrounds:
            x, y, w, h = fg.bbox.coordinates.astype(int)
            x = min(max(x, 0), display_image.shape[1] - 1)
            y = max(min(y, display_image.shape[0] - 1), 0)
            if hasattr(fg, 'mask'):
                resized_mask = cv2.resize(fg.mask, (w, h)) 

                colored_mask = cv2.applyColorMap((resized_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)

                mask_position = display_image[y:y+h, x:x+w]

                display_image[y:y+h, x:x+w] = cv2.addWeighted(mask_position, 0.5, colored_mask, 0.5, 0)
    
    def show_segmentation(self, display_image):
        for fg in self.foregrounds:
            if hasattr(fg, 'segmentation'):
                segmentation_adjusted = np.array(fg.segmentation, dtype=np.int32)
                segmentation_adjusted = segmentation_adjusted.reshape((-1, 1, 2))
                
                cv2.drawContours(display_image, [segmentation_adjusted], -1, (0, 255, 0), thickness=cv2.FILLED)

    def show_class(self, display_image):
        for fg in self.foregrounds:
            x, y, w, h = fg.bbox.coordinates.astype(int)
            x = min(max(x, 0), display_image.shape[1] - 1)
            y = max(min(y, display_image.shape[0] - 1), 0)
            if hasattr(fg, 'cls'):
                cv2.putText(display_image, fg.cls, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            3, (255, 255, 255), 2, cv2.LINE_AA)