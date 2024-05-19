class Filter:
    def apply(self, image):
        raise NotImplementedError('The apply method must be implemented by the subclass')
    def __str__(self) -> str:
        return self.__class__.__name__