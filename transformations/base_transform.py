from image_management import Object
class Transformation:
    def apply(self, image: Object):
        raise NotImplementedError('The apply method must be implemented by the subclass')