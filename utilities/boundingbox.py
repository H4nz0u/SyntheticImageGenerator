class BoundingBox:
    def __init__(self, coordinates, format_type="min_max"):
        self.coordinates = coordinates
        self.format_type = format_type

    def update_for_rotation(self, angle):
        # Update coordinates based on rotation
        pass

    def update_for_translation(self, dx, dy):
        # Update coordinates based on translation
        pass

    def update_for_scaling(self, scale):
        # Update coordinates based on scaling
        pass