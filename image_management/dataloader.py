import os
import random
import glob
from PIL import Image
from .object import ImgObject
from typing import Dict, List, Tuple, Union
from functools import lru_cache
import cv2
from utilities import logger

class DataLoader:
    def __init__(self, root_path: str, seed: int = None) -> None:
        self.root_path = root_path
    
    def _get_image_paths_for_type(self, path: str) -> List[str]:
        image_types = ["jpg", "png", "jpeg", "JPG"]
        return [file for ext in image_types for file in glob.glob(os.path.join(path, f"*.{ext}"))]

    def get_image(self) -> Union[ImgObject,cv2.typing.MatLike]:
        raise NotImplementedError
    
    def _serve_object(self, image_path: str) -> ImgObject:
        raise NotImplementedError

class ImageDataLoader(DataLoader):
    def __init__(self, root_path: str, objects: Dict[str, str], seed: int = None) -> None:
        super().__init__(root_path, seed)
        self.objects = {k: os.path.join(root_path, v) for k, v in objects.items()}
        self.object_image_paths = self._get_image_paths()

    def _get_image_paths(self) -> Tuple[List[str], Dict[str, List[str]]]:
        object_image_paths = {obj: self._get_image_paths_for_type(path) for obj, path in self.objects.items()}
        return object_image_paths
    
    #@lru_cache(maxsize=1024)
    def _serve_object(self, image_path: str) -> ImgObject:
        try:
            file_ending = image_path.split(".")[-1]
            annotation_path = image_path.replace(file_ending, "xml")
            obj = ImgObject(image_path, annotation_path)
            return obj
        except Exception as e:
            logger.error(f"Failed to construct object {image_path}: {str(e)}")
            raise e

    def get_image(self, image_type: str) -> ImgObject:
        if image_type in self.objects:
            path = random.choice(self.object_image_paths[image_type])
            logger.info(f"foreground: {path}")
            return self._serve_object(path)
        else:
            logger.error(f"Type {image_type} not found in objects")
            raise KeyError(f"Type {image_type} not found in objects")


class BackgroundDataLoader(DataLoader):
    def __init__(self, root_path: str, seed: int = None) -> None:
        super().__init__(root_path, seed)
        self.background_image_paths = self._get_image_paths_for_type(root_path)

    #@lru_cache(maxsize=1024)
    def _serve_background(self, image_path: str):
        return cv2.imread(image_path)

    def get_image(self) -> ImgObject:
        path = random.choice(self.background_image_paths)
        logger.info(f"background: {path}")
        return self._serve_background(path)