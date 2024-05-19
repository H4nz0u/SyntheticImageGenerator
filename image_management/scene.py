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
        obj_h, obj_w = foreground.image.shape[:2]
        background_h, background_w = self.background.shape[:2]

        # Calculate the center position of the foreground on the background
        x_start = int((background_w - obj_w) * position_x)
        y_start = int((background_h - obj_h) * position_y)

        # Clamping the start values to ensure they are within the background dimensions
        x_start = max(min(x_start, background_w - obj_w), 0)
        y_start = max(min(y_start, background_h - obj_h), 0)

        # Calculate the end points, ensuring they do not exceed the background dimensions
        x_end = min(x_start + obj_w, background_w)
        y_end = min(y_start + obj_h, background_h)

        # Adjusting the foreground image and mask dimensions if necessary
        cropped_foreground_width = x_end - x_start
        cropped_foreground_height = y_end - y_start

        # Resizing the mask and converting it to ensure it matches the cropped foreground dimensions
        mask = cv2.GaussianBlur(foreground.mask, (21, 21), 0)
        mask = cv2.resize(mask, (cropped_foreground_width, cropped_foreground_height))
        
        mask = mask.astype(np.float32) / 255.0  # Normalize mask to range 0-1 for blending
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if len(mask.shape) == 2 else mask
        
        # Crop the foreground image to the exact size of the region
        foreground_region = foreground.image[:cropped_foreground_height, :cropped_foreground_width]
        # Blend the foreground onto the background
        roi = self.background[y_start:y_end, x_start:x_end]
        self.background[y_start:y_end, x_start:x_end] = roi * (1 - mask) + foreground_region * mask

        # Update the bounding box coordinates to match the new position
        foreground.bbox.coordinates = np.array([x_start, y_start, cropped_foreground_width, cropped_foreground_height])
        foreground.mask = mask
        foreground.segmentation += np.array([x_start, y_start], dtype=np.int32)

        # Draw bounding box on the final image for testing
        """self.image = cv2.polylines(
            car_image_with_sign, [self.signImage.bounding_box], True, (0, 255, 0), 2
        )"""

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

    def show(self):
        # Create a copy of the background to draw on
        display_image = self.background.copy()
        
        for fg in self.foregrounds:
            x, y, w, h = fg.bbox.coordinates.astype(int)
            x = min(max(x, 0), display_image.shape[1] - 1)
            y = max(min(y, display_image.shape[0] - 1), 0)
            print(fg.bbox.coordinates.astype(int))
            # Draw bounding box
            cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw class name if available
            if hasattr(fg, 'cls'):
                cv2.putText(display_image, fg.cls, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Apply segmentation mask in a semi-transparent overlay
            if hasattr(fg, 'segmentation'):
                segmentation_adjusted = np.array(fg.segmentation, dtype=np.int32) + np.array([x, y])
                segmentation_adjusted = segmentation_adjusted.reshape((-1, 1, 2))  # Ensure correct shape
                
                # Draw the segmentation
                cv2.drawContours(display_image, [segmentation_adjusted], -1, (0, 255, 0), thickness=cv2.FILLED)
    
            
            # Overlay the foreground mask
            if hasattr(fg, 'mask'):
                # Resize the mask to exactly match the ROI dimensions
                resized_mask = cv2.resize(fg.mask, (w, h))  # Ensure mask is resized to (width, height) of the bounding box

                # Apply color map to the resized mask
                colored_mask = cv2.applyColorMap((resized_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)

                # Retrieve the actual position of the foreground in the background image
                mask_position = display_image[y:y+h, x:x+w]

                # Debug output to verify dimensions
                print(mask_position.shape, colored_mask.shape)  # These should now match

                # Blend the resized and colored mask with the background
                display_image[y:y+h, x:x+w] = cv2.addWeighted(mask_position, 0.5, colored_mask, 0.5, 0)


        # Show the final result
        cv2.imshow("Scene with Annotations", cv2.resize(display_image, (800, 600)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()