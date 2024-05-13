from pathlib import Path
from config import Config
from image_management import ImageDataLoader, Scene
from transformations import create_transformation, transformation_registry_dict
import cv2
import numpy as np
def main(data_config_path: Path, transformation_config_path: Path):
    config = Config(".", transformation_config_path, data_config_path)
    dataloader = ImageDataLoader(".", config["background_folder"], config["foreground_objects"], seed=config["seed"])
    image = dataloader.get_image("Typlabel-China")
    background = dataloader.get_image("background")
    for transformation in config["transformations"]["Typlabel-China"]:
        transformation.apply(image)
    x, y, w, h = image.bbox.coordinates
    
    scene = Scene(background, image)
    scene.add_foreground(image)
    
    cv2.imshow("image", cv2.resize(scene.background, (800, 600)))
    cv2.waitKey(0)    
    
if __name__ == "__main__":
    main('data_config.yaml', 'transformation_config.yaml')