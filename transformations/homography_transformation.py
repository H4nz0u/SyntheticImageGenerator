from .base_transform import Transformation
from utilities import register_transformation, logger, get_cached_dataframe
from image_management import Image, ImgObject
import cv2
import numpy as np


@register_transformation
class HomographyTransformation(Transformation):
    def __init__(self, homography_matrix, bbox_width=None, bbox_height=None):
        """
        Initialize the HomographyTransformation with a full 3x3 homography matrix
        and optional bounding box dimensions for resizing.
        
        :param homography_matrix: A 3x3 numpy array representing the homography.
        :param bbox_width: Optional. The target width for resizing the image.
        :param bbox_height: Optional. The target height for resizing the image.
        """
        if not isinstance(homography_matrix, np.ndarray) or homography_matrix.shape != (3, 3):
            raise ValueError("Homography matrix must be a 3x3 numpy array.")
        
        self.homography_matrix = homography_matrix.astype(np.float32)
        self.bbox_width = bbox_width
        self.bbox_height = bbox_height

        logger.info(f"Initialized HomographyTransformation with matrix:\n{self.homography_matrix}")
        if self.bbox_width and self.bbox_height:
            logger.info(f"Configured to resize images to width={self.bbox_width}, height={self.bbox_height}")

    def _calculate_new_dimensions(self, h, w, H):
        """
        Calculate the new image dimensions after applying the homography.
        
        :param h: Original image height.
        :param w: Original image width.
        :param H: Homography matrix (3x3).
        :return: new_width, new_height, min_x, min_y
        """
        # Original corners of the image in 2D
        corners = np.array([
            [0, 0],    # top-left
            [w, 0],    # top-right
            [w, h],    # bottom-right
            [0, h]     # bottom-left
        ], dtype=np.float32)
        
        # Convert corners to homogeneous coordinates
        corners_homogeneous = np.hstack([corners, np.ones((4, 1), dtype=np.float32)]).T  # Shape: (3,4)
        
        # Apply homography
        transformed_corners = H @ corners_homogeneous  # Shape: (3,4)
        transformed_corners /= transformed_corners[2, :]  # Normalize by the third row
        
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
        
        logger.info(f"Transformed corners: {transformed_corners}")
        logger.info(f"New dimensions after homography: width={new_width}, height={new_height}, min_x={min_x}, min_y={min_y}")
        
        return new_width, new_height, min_x, min_y

    def apply(self, obj: ImgObject):
        """
        Apply the homography transformation to the ImgObject, including resizing
        and shifting of the image, mask, segmentation, and bounding box.
        
        :param obj: ImgObject instance containing image, segmentation, mask, and bbox.
        """
        H = self.homography_matrix.copy()
        logger.info(f"Applying homography transformation with matrix:\n{H}")
        
        original_h, original_w = obj.image.shape[:2]
        logger.info(f"Original image dimensions: width={original_w}, height={original_h}")
        logger.info(f"Original bounding box coordinates: {obj.bbox.coordinates}")
        
        # Resize image and related attributes if bbox dimensions are provided
        if self.bbox_width and self.bbox_height:
            logger.info(f"Resizing image to width={self.bbox_width}, height={self.bbox_height}")
            
            # Calculate scaling factors
            scaling_factor_x = self.bbox_width / original_w
            scaling_factor_y = self.bbox_height / original_h
            logger.info(f"Scaling factors: x={scaling_factor_x}, y={scaling_factor_y}")
            
            # Resize image
            obj.image = cv2.resize(obj.image, (self.bbox_width, self.bbox_height), interpolation=cv2.INTER_LANCZOS4)
            logger.info(f"Resized image to ({self.bbox_width}, {self.bbox_height})")
            
            # Resize mask if it exists
            if obj.mask:
                if obj.mask.size > 0:
                    obj.mask = cv2.resize(obj.mask, (self.bbox_width, self.bbox_height), interpolation=cv2.INTER_NEAREST)
                    logger.info(f"Resized mask to ({self.bbox_width}, {self.bbox_height})")
                
            # Scale segmentation points
            if obj.segmentation is not None and len(obj.segmentation) > 0:
                obj.segmentation = self._scale_segmentation(obj.segmentation, scaling_factor_x, scaling_factor_y)
                logger.info(f"Scaled segmentation points by factors: x={scaling_factor_x}, y={scaling_factor_y}")
            
            # Scale bounding box coordinates
            if obj.bbox is not None and len(obj.bbox.coordinates) == 4:
                obj.bbox.coordinates = self._scale_bbox(obj.bbox.coordinates, scaling_factor_x, scaling_factor_y)
                logger.info(f"Scaled bounding box coordinates to: {obj.bbox.coordinates}")
        
        # Calculate new dimensions and the shift in coordinates
        h, w = obj.image.shape[:2]
        new_width, new_height, min_x, min_y = self._calculate_new_dimensions(h, w, H)
        logger.info(f"New image dimensions: width={new_width}, height={new_height}")
        logger.info(f"Shift in coordinates: min_x={min_x}, min_y={min_y}")
        
        # Translation to ensure the entire transformed image fits in the new canvas
        translation_matrix = np.array([
            [1, 0, -min_x],
            [0, 1, -min_y],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Adjust the homography matrix to account for translation
        H_adjusted = translation_matrix @ H  # Result is 3x3
        logger.debug(f"Adjusted Homography Matrix:\n{H_adjusted}")
        
        # Normalize the homography matrix to ensure H[2,2] = 1
        if H_adjusted[2, 2] != 0:
            H_adjusted /= H_adjusted[2, 2]
        else:
            logger.warning("H_adjusted[2,2] is zero. Skipping normalization.")
        
        # Apply the adjusted homography
        obj.image = cv2.warpPerspective(obj.image, H_adjusted, (new_width, new_height), flags=cv2.INTER_LANCZOS4)
        logger.info(f"Applied homography to image. New size: ({new_width}, {new_height})")
        
        # Transform other attributes if they exist
        if obj.segmentation.size > 0:
            self._transform_segmentation(obj, H_adjusted, new_width, new_height)
        if obj.mask:
            if obj.mask.size > 0:
                self._transform_mask(obj, H_adjusted, new_width, new_height)
        self._transform_bbox(obj, H_adjusted, new_width, new_height)

    def _transform_mask(self, obj, H, new_width, new_height):
        """
        Apply the homography to the mask.
        
        :param obj: ImgObject instance.
        :param H: Adjusted homography matrix (3x3).
        :param new_width: New image width.
        :param new_height: New image height.
        """
        mask = obj.mask
        if len(mask.shape) == 3 and mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        transformed_mask = cv2.warpPerspective(mask, H, (new_width, new_height), flags=cv2.INTER_NEAREST)
        obj.mask = transformed_mask
        logger.debug("Mask transformed successfully.")
    
    def _transform_segmentation(self, obj, H, new_width, new_height):
        """
        Apply the homography to the segmentation.
        
        :param obj: ImgObject instance.
        :param H: Adjusted homography matrix (3x3).
        :param new_width: New image width.
        :param new_height: New image height.
        """
        segmentation = np.array(obj.segmentation, dtype=np.float32)
        logger.debug(f"Original segmentation data (flattened): {segmentation}")
        logger.debug(f"Number of segmentation elements: {len(segmentation)}")
        
        segmentation = segmentation.reshape((-1, 1, 2))
        logger.debug(f"Reshaped segmentation data: {segmentation.shape}")
        
        # Apply homography
        transformed_segmentation = cv2.perspectiveTransform(segmentation, H).reshape(-1, 2)
        logger.debug(f"Transformed segmentation data shape: {transformed_segmentation.shape}")
        logger.debug(f"Transformed segmentation data: {transformed_segmentation}")
        
        # Clip the segmentation points to be within the new image bounds
        transformed_segmentation[:, 0] = np.clip(transformed_segmentation[:, 0], 0, new_width - 1)
        transformed_segmentation[:, 1] = np.clip(transformed_segmentation[:, 1], 0, new_height - 1)
        
        obj.segmentation = transformed_segmentation.astype(np.int32)
        logger.debug("Segmentation transformed successfully.")
    
    def _transform_bbox(self, obj, H, new_width, new_height):
        """
        Update the bounding box based on the transformed mask.
        
        :param obj: ImgObject instance.
        :param H: Adjusted homography matrix (3x3).
        :param new_width: New image width.
        :param new_height: New image height.
        """
        obj.bbox.coordinates = self.update_bbox_from_mask(obj.mask)
        logger.debug("Bounding box updated successfully.")
    
    def _scale_segmentation(self, segmentation, scale_x, scale_y):
        """
        Scale segmentation points by the given factors while maintaining the (N, 1, 2) shape.

        :param segmentation: NumPy array of segmentation points with shape (N, 1, 2).
        :param scale_x: Scaling factor for x-coordinates.
        :param scale_y: Scaling factor for y-coordinates.
        :return: Scaled segmentation points with shape (N, 1, 2).
        """
        try:
            # Ensure segmentation is a NumPy array with shape (N, 1, 2)
            segmentation_array = np.array(segmentation, dtype=np.float32)
            if len(segmentation_array.shape) != 3:
                segmentation_scaled = segmentation_array.copy()
                segmentation_scaled[:, 0] *= scale_x
                segmentation_scaled[:, 1] *= scale_x
            else:
            # Apply scaling factors
                segmentation_scaled = segmentation_array.copy()
                segmentation_scaled[:, 0, 0] *= scale_x  # Scale x-coordinates
                segmentation_scaled[:, 0, 1] *= scale_y  # Scale y-coordinates

            logger.debug(f"Scaled segmentation points: {segmentation_scaled}")
            return segmentation_scaled

        except Exception as e:
            logger.error(f"Error scaling segmentation points: {e}")
            raise


    def _scale_bbox(self, bbox, scale_x, scale_y):
        """
        Scale bounding box coordinates by the given factors.

        :param bbox: List or array [x_min, y_min, x_max, y_max].
        :param scale_x: Scaling factor for x-coordinates.
        :param scale_y: Scaling factor for y-coordinates.
        :return: Scaled bounding box coordinates.
        """
        try:
            bbox_array = np.array(bbox, dtype=np.float32)
            bbox_scaled = bbox_array.copy()
            bbox_scaled[0] *= scale_x  # x_min
            bbox_scaled[1] *= scale_y  # y_min
            bbox_scaled[2] *= scale_x  # x_max
            bbox_scaled[3] *= scale_y  # y_max

            bbox_scaled = bbox_scaled.tolist()
            logger.debug(f"Scaled bounding box: {bbox_scaled}")
            return bbox_scaled

        except Exception as e:
            logger.error(f"Error scaling bounding box: {e}")
            raise


@register_transformation
class HomographyTransformationFromDataFrame(HomographyTransformation):
    def __init__(self, dataframe_path="standard", homography_column="H", class_column="class", 
                 bbox_width_column="bbox_width", bbox_height_column="bbox_height"):
        """
        Initialize the HomographyTransformationFromDataFrame with a dataframe path.

        :param dataframe_path: Path to the dataframe containing homography matrices and bbox dimensions.
        :param homography_column: Column name where homography matrices are stored.
        :param class_column: Column name used to filter homography matrices based on class.
        :param bbox_width_column: Column name for bounding box width.
        :param bbox_height_column: Column name for bounding box height.
        """
        self.dataframe_path = dataframe_path
        self.homography_column = homography_column
        self.class_column = class_column
        self.bbox_width_column = bbox_width_column
        self.bbox_height_column = bbox_height_column
        self.data = get_cached_dataframe(self.dataframe_path)

        # Initialize with an identity matrix and no resizing
        identity_homography = np.eye(3, dtype=np.float32)
        super().__init__(identity_homography)
        logger.info("Initialized HomographyTransformationFromDataFrame with identity matrix.")

    def _select_homography_matrix_and_bbox(self, cls):
        """
        Select a homography matrix and bbox dimensions from the dataframe based on the class.

        :param cls: Class label of the image object.
        :return: Tuple containing homography matrix, bbox_width, and bbox_height.
        """
        try:
            # Filter the DataFrame by the class
            filtered_data = self.data[self.data[self.class_column] == cls]
            if filtered_data.empty:
                raise ValueError(f"No data found for class '{cls}' in DataFrame.")

            # Sample one row randomly
            selected_row = filtered_data.sample(n=1).iloc[0]

            # Extract the homography matrix from the row
            homography_str = selected_row[self.homography_column]
            logger.info(f"Selected homography matrix from Image '{selected_row.get('image_name', 'unknown')}'.")

            # Parse the homography matrix
            if isinstance(homography_str, str):
                # Example format: "1,0,0,0,1,0,0,0,1"
                homography_values = list(map(float, homography_str.split(',')))
                if len(homography_values) != 9:
                    raise ValueError(f"Invalid homography matrix length: {len(homography_values)}. Expected 9 elements.")
            elif isinstance(homography_str, (list, np.ndarray)):
                homography_values = homography_str
                if np.array(homography_values).shape != (3, 3):
                    raise ValueError(f"Invalid homography matrix shape: {np.array(homography_values).shape}. Expected (3, 3).")
            else:
                raise ValueError("Homography column must contain a string or list/array of numbers.")

            homography_matrix = np.array(homography_values, dtype=np.float32).reshape((3, 3))

            # Normalize the homography matrix
            if homography_matrix[2, 2] != 0:
                homography_matrix /= homography_matrix[2, 2]
            else:
                logger.warning("Homography matrix normalization skipped due to H[2,2] being zero.")

            logger.info(f"Selected homography matrix for class '{cls}':\n{homography_matrix}")

            # Extract bbox dimensions
            bbox_width = selected_row.get(self.bbox_width_column, None)
            bbox_height = selected_row.get(self.bbox_height_column, None)

            if bbox_width is None or bbox_height is None:
                raise ValueError(f"BBox dimensions not found in columns '{self.bbox_width_column}' and '{self.bbox_height_column}'.")

            logger.info(f"Extracted bbox_width={bbox_width}, bbox_height={bbox_height} for class '{cls}'.")
            
            return homography_matrix, bbox_width, bbox_height

        except Exception as e:
            logger.error(f"Failed to select homography matrix and bbox for class '{cls}': {e}")
            raise

    def apply(self, obj: ImgObject):
        """
        Apply the homography transformation based on the class label.
        This includes selecting the appropriate homography matrix and bbox dimensions
        from the dataframe, then delegating to the base class for resizing and transformation.

        :param obj: ImgObject instance containing image, segmentation, mask, and bbox.
        """
        # Select homography matrix and bbox dimensions
        homography_matrix, bbox_width, bbox_height = self._select_homography_matrix_and_bbox(obj.cls)
        
        # Update the homography matrix and bbox dimensions in the base class
        self.homography_matrix = homography_matrix
        self.bbox_width = bbox_width
        self.bbox_height = bbox_height

        logger.info(f"Configured HomographyTransformationFromDataFrame with bbox_width={bbox_width}, bbox_height={bbox_height}")

        # Proceed with the homography transformation using the base class method
        super().apply(obj)
