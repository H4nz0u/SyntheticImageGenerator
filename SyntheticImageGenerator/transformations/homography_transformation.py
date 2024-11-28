from .base_transform import Transformation
from ..utilities import register_transformation, logger, get_cached_dataframe
from ..image_management import Image, ImgObject
import cv2
import numpy as np


@register_transformation
class HomographyTransformation(Transformation):
    def __init__(self, homography_matrix):
        """
        Initialize the HomographyTransformation with a full 3x3 homography matrix
        and optional bounding box dimensions for resizing.
        
        :param homography_matrix: A 3x3 numpy array representing the homography.
        :param bbox_width: Optional. The target width for resizing the image.
        :param bbox_height: Optional. The target height for resizing the image.
        """
        if isinstance(homography_matrix, list):
            homography_matrix = np.array(homography_matrix, dtype=np.float32)
        elif not isinstance(homography_matrix, np.ndarray):
            raise ValueError("Homography matrix must be a list of lists or a 3x3 numpy array.")

        if homography_matrix.shape != (3, 3):
            raise ValueError("Homography matrix must be a 3x3 structure.")
        
        self.homography_matrix = homography_matrix.astype(np.float32)

        logger.info(f"Initialized HomographyTransformation with matrix:\n{self.homography_matrix}")

    
    def apply(self, obj: ImgObject):
        """
        Apply the homography transformation to the image and adapt the bbox, segmentation, and mask.
        
        :param obj: ImgObject instance.
        """
        # Read the image
        image = obj.image
        height, width = image.shape[:2]

        # Prepare points to transform: corners of the image
        corners = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32).reshape(-1, 1, 2)  # Shape (4, 1, 2)

        # Transform the corners
        transformed_corners = cv2.perspectiveTransform(corners, self.homography_matrix)
        transformed_corners = transformed_corners.reshape(-1, 2)

        # Compute bounding rectangle of the transformed image
        x_coords = transformed_corners[:, 0]
        y_coords = transformed_corners[:, 1]
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        # Compute the translation to shift the image such that the top-left corner is at (0,0)
        tx = -x_min if x_min < 0 else 0
        ty = -y_min if y_min < 0 else 0
        # Create translation matrix
        translation_matrix = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ], dtype=np.float32)
        # Adjusted homography
        adjusted_homography = translation_matrix @ self.homography_matrix

        # Compute the size of the output image
        new_width = int(np.ceil(x_max + tx))
        new_height = int(np.ceil(y_max + ty))

        # Apply the homography to the image
        transformed_image = cv2.warpPerspective(image, adjusted_homography, (new_width, new_height))
        # Update the image in the ImgObject
        obj.image = transformed_image

        # Now apply the homography to the mask, segmentation, and bounding box
        self._transform_mask(obj, adjusted_homography, new_width, new_height)
        self._transform_segmentation(obj, adjusted_homography, new_width, new_height)
        self._transform_bbox(obj, adjusted_homography, new_width, new_height)



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
        if obj.mask is None or obj.mask.size == 0:
            logger.warning("Mask is None or empty. Skipping bounding box update.")
            return
        obj.bbox.coordinates = self.update_bbox_from_mask(obj.mask)
        logger.debug("Bounding box updated successfully.")
    


@register_transformation
class HomographyTransformationFromDataFrame(HomographyTransformation):
    def __init__(self, dataframe_path="standard", homography_column="transformation_matrix", class_column="class"):
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
            homography_str = self.data.sample_parameter(self.homography_column, filters={self.class_column: cls})
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
            
            return homography_matrix

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
        homography_matrix = self._select_homography_matrix_and_bbox(obj.cls)
        
        # Update the homography matrix and bbox dimensions in the base class
        self.homography_matrix = homography_matrix

        # Proceed with the homography transformation using the base class method
        super().apply(obj)
