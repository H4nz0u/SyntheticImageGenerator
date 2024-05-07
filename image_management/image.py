import cv2

class Image:
    def __init__(self, path):
        self.path = path
        self.image = cv2.imread(path)
    def show(self):
        cv2.imshow(self.path, self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def save(self, path):
        cv2.imwrite(path, self.image)
    def get_image(self):
        return self.image
    def get_path(self):
        return self.path
    def get_shape(self):
        return self.image.shape
    