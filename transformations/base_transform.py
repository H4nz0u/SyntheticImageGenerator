from image_management import ImgObject
class Transformation:
    def apply(self, image: ImgObject):
        raise NotImplementedError('The apply method must be implemented by the subclass')
    def __str__(self) -> str:
        return self.__class__.__name__