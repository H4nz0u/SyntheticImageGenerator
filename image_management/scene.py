import cv2
from image_management import ImgObject
import numpy as np
from object_position import BasePositionDeterminer
from annotations import BaseAnnotator
import os
from typing import List
from pathlib import Path
from utilities import logger
from filter.brightness import TargetBrightness
from filter import Filter
class Scene:
    def __init__(self, background) -> None:
        self.background = background
        self.foregrounds: List[ImgObject] = []
        self.filters: List[Filter] = []
    
    def add_filter(self, filter):
        self.filters.append(filter)
    
    def apply_filter(self):
        for filter in self.filters:
            if isinstance(filter, TargetBrightness):
                self.background = filter.apply(self.background, self.foregrounds[0].bbox.coordinates)
            else:
                self.background = filter.apply(self.background)
            
    def add_foreground(self, foreground: ImgObject):
        self.foregrounds.append(foreground)
        x_start, y_start, x_end, y_end = self.positionDeterminer.get_position(self.background, self.foregrounds)        

        # Clipping dimensions if necessary
        target_width = x_end - x_start
        target_height = y_end - y_start

        # Use clipped image regions
        clipped_image = cv2.resize(foreground.image, (target_width, target_height))
        if foreground.mask is not None:
            clipped_mask = cv2.resize(foreground.mask, (target_width, target_height))

            # Normalize and prepare mask for blending
            mask = clipped_mask.astype(np.float32) / 255.0
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if len(mask.shape) == 2 else mask

            # Retrieve ROI from the background
            roi = self.background[y_start:y_end, x_start:x_end]

            # Blend the clipped foreground onto the background
            self.background[y_start:y_end, x_start:x_end] = roi * (1 - mask) + clipped_image * mask

            foreground.mask = mask
        else:
            # Simply place the clipped image if no mask is provided
            self.background[y_start:y_end, x_start:x_end] = clipped_image

        foreground.bbox.coordinates = np.array([x_start, y_start, target_width, target_height])
        if foreground.segmentation is not None:
            scale_x = target_width / foreground.image.shape[1]
            scale_y = target_height / foreground.image.shape[0]
            scale_matrix = np.array([scale_x, scale_y])
            foreground.segmentation = (foreground.segmentation * scale_matrix).astype(int)
            foreground.segmentation += np.array([x_start, y_start])
            
        self.add_annotation(foreground)


    def configure_positioning(self, positionDeterminer: BasePositionDeterminer):
        self.positionDeterminer = positionDeterminer

    def configure_annotator(self, annotator: BaseAnnotator):
        self.annotator = annotator
        self.annotator.reset()
    
    def add_annotation(self, obj):
        if obj.segmentation.size > 0:
            self.annotator.append_object(obj.segmentation, obj.cls)
        else:
            c = obj.bbox.coordinates
            bbox_coordinates = np.array([(c[0],c[1]), (c[0]+c[2], c[1]), (c[0]+c[2], c[1]+c[3]), (c[0], c[1]+c[3])])
            self.annotator.append_object(bbox_coordinates, obj.cls)

    def write(self, path: Path, size, annotation=True):
        image = cv2.resize(self.background, size)
        if annotation:
            if not path.parent.exists():
                os.makedirs(path.parent)
            filename = os.path.basename(path)
            file_ending = filename.split(".")[-1]
            # Replace the file ending with xml
            xml_path = path.with_name(filename.replace(file_ending, "xml"))                
            self.annotator.write_xml(xml_path, image.shape)
        cv2.imwrite(path.as_posix(), image)

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

        #cv2.imshow("Scene with Annotations", cv2.resize(display_image, (800, 600)))
        cv2.imwrite("test_visualization.jpg", cv2.resize(display_image, (2000, 1500)))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

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
            if fg.mask:
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