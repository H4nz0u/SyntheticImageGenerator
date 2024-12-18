class BoundingBox:
    def __init__(self, coordinates, format_type="min_max"):
        self.coordinates = coordinates
        self.format_type = format_type
    
    def area(self):
        if self.format_type == "min_max":
            xmin, ymin, xmax, ymax = self.coordinates
            width = xmax - xmin
            height = ymax - ymin
        elif self.format_type == "center":
            # If the format is center-based, the coordinates might be [center_x, center_y, width, height]
            center_x, center_y, width, height = self.coordinates
        else:
            raise ValueError("Unsupported format_type for bounding box")

        return max(0, width) * max(0, height)

    def __str__(self):
        return f"BoundingBox({self.coordinates})"