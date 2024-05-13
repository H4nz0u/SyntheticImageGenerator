from image_management import ImgObject
class Transformation:
    def apply(self, image: ImgObject):
        raise NotImplementedError('The apply method must be implemented by the subclass')