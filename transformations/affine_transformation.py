from .base_transform import Transformation
from utilities import register_transformation, logger, get_cached_dataframe
from image_management import Image, ImgObject
import cv2
import numpy as np

@register_transformation
class AffineTransformation(Transformation):
    def __init__(self, angle, scale_x, scale_y, shear_x, shear_y):
        self.rotation_angle = angle
        angle_rad = np.radians(angle)
        self.rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_matrix = np.array([
            [scale_x, 0, 0],
            [0, scale_y, 0],
            [0, 0, 1]
        ])
        self.shear_x = shear_x
        self.shear_x_matrix = np.array([
            [1, shear_x, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        self.shear_y = shear_y
        self.shear_y_matrix = np.array([
            [1, 0, 0],
            [shear_y, 1, 0],
            [0, 0, 1]
        ])

    def _calculate_new_dimensions(self, h, w, M):
        # Original corners of the image in 2D
        corners = np.array([
            [0, 0],    # top-left
            [w, 0],    # top-right
            [0, h],    # bottom-left
            [w, h]     # bottom-right
        ])
        
        # Convert M to a full 3x3 matrix to apply it to the corners
        M_full = np.vstack([M, [0, 0, 1]])  # Convert to 3x3 by adding [0, 0, 1]
        
        # Apply the transformation matrix to the corners
        transformed_corners = M_full @ np.hstack([corners, np.ones((4, 1))]).T
        
        # Extract the transformed x and y coordinates
        transformed_x = transformed_corners[0, :]
        transformed_y = transformed_corners[1, :]
        
        # Calculate new bounding box
        min_x = np.min(transformed_x)
        max_x = np.max(transformed_x)
        min_y = np.min(transformed_y)
        max_y = np.max(transformed_y)
        
        # New dimensions
        new_width = int(np.ceil(max_x - min_x))
        new_height = int(np.ceil(max_y - min_y))
        
        return new_width, new_height, min_x, min_y

    def apply(self, obj: ImgObject):
        # Composite transformation matrix
        M = (self.rotation_matrix @ self.scale_matrix @ self.shear_x_matrix @ self.shear_y_matrix)[:2, :]
        logger.info(f"Applying affine transformation with angle={self.rotation_angle}, scale_x={self.scale_x}, scale_y={self.scale_y}, shear_x={self.shear_x}, shear_y={self.shear_y}")
        # Calculate new dimensions and the shift in coordinates
        new_width, new_height, min_x, min_y = self._calculate_new_dimensions(obj.image.shape[0], obj.image.shape[1], M)
        
        # Adjust the transformation matrix to account for the translation
        translation_matrix = np.array([[1, 0, -min_x],
                                    [0, 1, -min_y]])
        
        # Convert translation_matrix to 3x3 to multiply with M_full
        M_full = np.vstack([M, [0, 0, 1]])
        M_adjusted_full = translation_matrix @ M_full
        
        # Convert back to 2x3 for cv2.warpAffine
        M_adjusted = M_adjusted_full[:2, :]
        
        # Apply the adjusted transformation
        #use cv2.INTER_NEAREST
        obj.image = cv2.warpAffine(obj.image, M_adjusted, (new_width, new_height), flags=cv2.INTER_LANCZOS4)
        
        if obj.segmentation.size > 0:
            self._transform_segmentation(obj, M_adjusted, new_width, new_height)
        if obj.mask:
            if obj.mask.size > 0:
                self._transform_mask(obj, M_adjusted, new_height, new_width)
        self._transform_bbox(obj, M_adjusted)



    def _transform_mask(self, obj, M, new_height, new_width):
        mask = obj.mask
        if len(mask.shape) == 3 and mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        rotated_mask = cv2.warpAffine(mask, M, (new_width, new_height))
        obj.mask = rotated_mask
    
    def _transform_segmentation(self, obj, M, new_width, new_height):
        segmentation = np.array(obj.segmentation, dtype=np.float32).reshape((-1, 1, 2))
        
        transformed_segmentation = cv2.transform(segmentation, M).reshape((-1, 2))
        
        transformed_segmentation[:, 0] = np.clip(transformed_segmentation[:, 0], 0, new_width)
        transformed_segmentation[:, 1] = np.clip(transformed_segmentation[:, 1], 0, new_height)

        obj.segmentation = transformed_segmentation.astype(np.int32)
    
    def _transform_bbox(self, obj, M):
        obj.bbox.coordinates = self.update_bbox_from_mask(obj.mask)

@register_transformation
class RandomAffineTransformation(AffineTransformation):
    def __init__(self, angle_range, scale_range, shear_range):
        self.angle_range = angle_range
        self.scale_range = scale_range
        self.shear_range = shear_range
        angle, shear_x, shear_y, scale_x, scale_y = self.get_random_values()
        super().__init__(angle, scale_x, scale_y, shear_x, shear_y)

    def apply(self, obj: ImgObject):
        angle, shear_x, shear_y, scale_x, scale_y = self.get_random_values()
        super().__init__(angle, scale_x, scale_y, shear_x, shear_y)
        super().apply(obj)

    def get_random_values(self):
        angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        scale_x = np.random.uniform(self.scale_range[0], self.scale_range[1])
        scale_y = np.random.uniform(self.scale_range[0], self.scale_range[1])
        shear_x = np.random.uniform(self.shear_range[0], self.shear_range[1])
        shear_y = np.random.uniform(self.shear_range[0], self.shear_range[1])
        return angle, shear_x, shear_y, scale_x, scale_y

@register_transformation
class AffineTransformationFromDataFrame(AffineTransformation):
    def __init__(self, dataframe_path, angle_column="angle", scale_x_column="scale_x", scale_y_column="scale_y", shear_x_column="shear_x", shear_y_column="shear_y"):
        column_names = {
            "angle": angle_column,
            "scale_x": scale_x_column,
            "scale_y": scale_y_column,
            "shear_x": shear_x_column,
            "shear_y": shear_y_column
        }
        self.dataframe_path = dataframe_path
        self.column_names = column_names
        self.data = get_cached_dataframe(self.dataframe_path)
        
        # Initialize with dummy values
        super().__init__(angle=0, scale_x=1, scale_y=1, shear_x=0, shear_y=0)

    def _select_transformation_params(self, cls):
        try:
            # Filter the DataFrame by the class
            filtered_data = self.data[self.data['class'] == cls]
            if filtered_data.empty:
                raise ValueError(f"No data found for class '{cls}' in DataFrame.")
            
            # Sample one row randomly
            selected_row = filtered_data.sample(n=1).iloc[0]
            
            # Extract transformation parameters from the row
            angle = selected_row[self.column_names["angle"]]
            scale_x = selected_row[self.column_names["scale_x"]]
            scale_y = selected_row[self.column_names["scale_y"]]
            shear_x = selected_row[self.column_names["shear_x"]]
            shear_y = selected_row[self.column_names["shear_y"]]
            logger.info(f"Selected transformation parameters for class '{cls}': angle={angle}, scale_x={scale_x}, scale_y={scale_y}, shear_x={shear_x}, shear_y={shear_y}")
            return angle, scale_x, scale_y, shear_x, shear_y
        except Exception as e:
            logger.error(f"Failed to select transformation parameters for class '{cls}': {e}")
            raise
    
    def apply(self, obj: ImgObject):
        angle, scale_x, scale_y, shear_x, shear_y = self._select_transformation_params(obj.cls)
        super().__init__(angle, scale_x, scale_y, shear_x, shear_y)
        super().apply(obj)