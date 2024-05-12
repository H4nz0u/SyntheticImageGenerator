import os
from utilities import random
import glob
from PIL import Image
from typing import Dict, List
from functools import lru_cache

class ImageDataLoader:
    def __init__(self, root_path: str, background_path: str, objects: Dict[str, str], seed: int = None) -> None:
        self.root_path = root_path
        self.background_path = os.path.join(root_path, background_path)
        self.objects = {k: os.path.join(root_path, v) for k, v in objects.items()}
        self.background_image_paths, self.object_image_paths = self._get_image_paths()

    def _get_image_paths_for_type(self, path: str) -> List[str]:
        image_types = ["jpg", "png", "jpeg"]
        return [file for ext in image_types for file in glob.glob(os.path.join(path, f"*.{ext}"))]

    def _get_image_paths(self) -> (List[str], Dict[str, List[str]]):
        background_image_paths = self._get_image_paths_for_type(self.background_path)
        object_image_paths = {obj: self._get_image_paths_for_type(path) for obj, path in self.objects.items()}
        return background_image_paths, object_image_paths
    
    @lru_cache(maxsize=1024)
    def _serve_image(self, image_path: str) -> Image.Image:
        try:
            with Image.open(image_path) as img:
                return img.copy()
        except Exception as e:
            print(f"Failed to load image {image_path}: {str(e)}")
            return None

    def get_image(self, image_type: str) -> Image.Image:
        if image_type == "background":
            path = random.choice(self.background_image_paths)
            return self._serve_image(path)
        elif image_type in self.objects:
            path = random.choice(self.object_image_paths[image_type])
            return self._serve_image(path)
        else:
            raise KeyError(f"Type {image_type} not found in objects")