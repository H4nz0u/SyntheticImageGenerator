from .base_transform import Transformation
from ..utilities import register_transformation, logger, get_cached_dataframe
from ..image_management import ImgObject
import cv2
import numpy as np
import math

@register_transformation
class AffineTransformation(Transformation):
    def __init__(self, angle, scale_x, scale_y, shear, shear_direction='x'):
        """
        Initialize the AffineTransformation.

        :param angle: Rotation angle in degrees.
        :param scale_x: Scaling factor along the x-axis.
        :param scale_y: Scaling factor along the y-axis.
        :param shear: Shear factor.
        :param shear_direction: Direction of shear, either 'x' or 'y'.
        """
        if shear_direction.lower() not in ['x', 'y']:
            raise ValueError("shear_direction must be either 'x' or 'y'.")
        
        self.angle = angle
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.shear = shear
        self.shear_direction = shear_direction.lower()
        
        # Construct the affine transformation matrix
        self.M = self._construct_affine_matrix()
    
    def _construct_affine_matrix(self):
        """
        Constructs a 2x3 affine transformation matrix from rotation, shear, and scale parameters.

        :return: 2x3 affine transformation matrix.
        """
        # Convert rotation from degrees to radians
        rotation_rad = math.radians(self.angle)
        cos_theta = math.cos(rotation_rad)
        sin_theta = math.sin(rotation_rad)

        # Construct Scaling Matrix (S)
        S = np.array([
            [self.scale_x, 0, 0],
            [0, self.scale_y, 0],
            [0, 0, 1]
        ])

        # Construct Shearing Matrix (Sh) based on direction
        if self.shear_direction == 'x':
            Sh = np.array([
                [1, self.shear, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
        else:  # 'y' direction
            Sh = np.array([
                [1, 0, 0],
                [self.shear, 1, 0],
                [0, 0, 1]
            ])

        # Construct Rotation Matrix (R)
        R = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])

        # Composite transformation: Rotation * Shear * Scale
        M = S @ Sh @ R

        # Return the 2x3 affine transformation matrix
        return M[:2, :]
    
    def _calculate_new_dimensions(self, h, w, M):
        """
        Calculates the new image dimensions after applying the affine transformation.

        :param h: Original image height.
        :param w: Original image width.
        :param M: Affine transformation matrix.
        :return: Tuple of (new_width, new_height, min_x, min_y).
        """
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
        
        # New dimensions with padding
        padding = 2  # Add 2 pixels of padding
        new_width = int(np.ceil(max_x - min_x)) + padding
        new_height = int(np.ceil(max_y - min_y)) + padding
        
        return new_width, new_height, min_x, min_y

    def apply(self, obj: ImgObject):
        """
        Applies the affine transformation to the ImgObject.

        :param obj: ImgObject containing image, mask, segmentation, and bounding box.
        """
        logger.info(f"Applying affine transformation with angle={self.angle}, "
                    f"scale_x={self.scale_x}, scale_y={self.scale_y}, "
                    f"shear={self.shear}, shear_direction={self.shear_direction.upper()}")
        
        # Calculate new dimensions and the shift in coordinates
        new_width, new_height, min_x, min_y = self._calculate_new_dimensions(
            obj.image.shape[0], obj.image.shape[1], self.M
        )
        # Adjust the transformation matrix to account for the translation
        translation_matrix = np.array([
            [1, 0, -min_x],
            [0, 1, -min_y],
            [0, 0, 1]
        ])
        
        # Convert M to a full 3x3 matrix
        M_full = np.vstack([self.M, [0, 0, 1]])
        M_adjusted_full = translation_matrix @ M_full
        # Convert back to 2x3 for cv2.warpAffine
        M_adjusted = M_adjusted_full[:2, :]

        # Choose interpolation method
        interpolation = cv2.INTER_NEAREST if obj.mask is not None else cv2.INTER_LANCZOS4
        # Apply the adjusted transformation to the image
        transformed_image = cv2.warpAffine(obj.image, M_adjusted, (new_width, new_height), flags=interpolation)
        obj.image = transformed_image
        
        # Apply the adjusted transformation to the mask if it exists
        if obj.mask is not None and obj.mask.size > 0:
            transformed_mask = self._transform_mask(obj, M_adjusted, new_width, new_height)
            obj.mask = transformed_mask
        cv2.imwrite("transformed_image.jpg", obj.image)
        # Apply the adjusted transformation to the segmentation if it exists
        if obj.segmentation.size > 0:
            transformed_segmentation = self._transform_segmentation(obj, M_adjusted, new_width, new_height)
            obj.segmentation = transformed_segmentation
        
        # Update the bounding box
        self._transform_bbox(obj)
        
        logger.info(f'New BBox coordinates: {obj.bbox.coordinates}')

    def _transform_mask(self, obj, M, new_width, new_height):
        """
        Transforms the mask of the ImgObject.

        :param obj: ImgObject containing the mask.
        :param M: Adjusted affine transformation matrix.
        :param new_width: New image width after transformation.
        :param new_height: New image height after transformation.
        :return: Transformed mask.
        """
        mask = obj.mask
        if len(mask.shape) == 3 and mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        rotated_mask = cv2.warpAffine(mask, M, (new_width, new_height), flags=cv2.INTER_NEAREST)
        return rotated_mask

    def _transform_segmentation(self, obj, M, new_width, new_height):
        """
        Transforms the segmentation points of the ImgObject.

        :param obj: ImgObject containing the segmentation.
        :param M: Adjusted affine transformation matrix.
        :param new_width: New image width after transformation.
        :param new_height: New image height after transformation.
        :return: Transformed segmentation points.
        """
        segmentation = np.array(obj.segmentation, dtype=np.float32).reshape((-1, 1, 2))
        transformed_segmentation = cv2.transform(segmentation, M).reshape((-1, 2))
        
        # Clip the segmentation points to lie within the new image boundaries
        transformed_segmentation[:, 0] = np.clip(transformed_segmentation[:, 0], 0, new_width - 1)
        transformed_segmentation[:, 1] = np.clip(transformed_segmentation[:, 1], 0, new_height - 1)


        return transformed_segmentation.astype(np.int32)

    def _transform_bbox(self, obj):
        """
        Updates the bounding box coordinates based on the transformed mask.

        :param obj: ImgObject containing the mask and bounding box.
        """
        obj.bbox.coordinates = self.update_bbox_from_mask(obj.mask)


@register_transformation
class RandomAffineTransformation(AffineTransformation):
    def __init__(self, angle_range, scale_range, shear_range, shear_direction='x'):
        """
        Initialize the RandomAffineTransformation.

        :param angle_range: Tuple of (min_angle, max_angle) in degrees.
        :param scale_range: Tuple of (min_scale, max_scale) for both x and y axes.
        :param shear_range: Tuple of (min_shear, max_shear).
        :param shear_direction: Direction of shear, either 'x' or 'y'.
        """
        if shear_direction.lower() not in ['x', 'y']:
            raise ValueError("shear_direction must be either 'x' or 'y'.")
        
        self.angle_range = angle_range
        self.scale_range = scale_range
        self.shear_range = shear_range
        self.shear_direction = shear_direction.lower()
        
        # Initialize with random parameters
        angle, shear, scale_x, scale_y = self.get_random_values()
        super().__init__(angle, scale_x, scale_y, shear, self.shear_direction)
    
    def get_random_values(self):
        """
        Generates random affine transformation parameters within the specified ranges.

        :return: Tuple of (angle, shear, scale_x, scale_y).
        """
        angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        scale_x = np.random.uniform(self.scale_range[0], self.scale_range[1])
        scale_y = np.random.uniform(self.scale_range[0], self.scale_range[1])
        shear = np.random.uniform(self.shear_range[0], self.shear_range[1])
        return angle, shear, scale_x, scale_y
    
    def apply(self, obj: ImgObject):
        """
        Applies a randomly generated affine transformation to the ImgObject.

        :param obj: ImgObject containing image, mask, segmentation, and bounding box.
        """
        # Generate new random parameters
        angle, shear, scale_x, scale_y = self.get_random_values()
        
        # Update the transformation parameters
        self.angle = angle
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.shear = shear
        
        # Reconstruct the affine matrix with new parameters
        self.M = self._construct_affine_matrix()
        
        # Apply the affine transformation
        super().apply(obj)

@register_transformation
class AffineTransformationFromDataFrame(AffineTransformation):
    def __init__(self, dataframe_path, column_name: str = "transformation_matrix"):
        self.dataframe_path = dataframe_path
        self.column_name = column_name
        self.data = get_cached_dataframe(self.dataframe_path)
        
        # Initialize with dummy values
        super().__init__(angle=0, scale_x=1, scale_y=1, shear=0)

    def _select_transformation_params(self, cls):
        try:
            # Filter the DataFrame by the class
            affine_transformation = self.data.sample_parameter(self.column_name, {"class": cls})
            return affine_transformation
        except Exception as e:
            logger.error(f"Failed to select transformation parameters for class '{cls}': {e}")
            raise
    
    def apply(self, obj: ImgObject):
        affine_transformation = self._select_transformation_params(obj.cls)
        self.M = np.array(affine_transformation)[:2, :]
        super().apply(obj)