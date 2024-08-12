from pathlib import Path
from config import Config
from image_management import ImageDataLoader, Scene, BackgroundDataLoader, ImgObject
import cv2
import numpy as np
from utilities import logger
import faulthandler
faulthandler.enable()
import cProfile
import pstats
import io

def generate_image(config: Config):
    dataloader = ImageDataLoader(".",  config["foreground_objects"], seed=config["seed"])
    background_dataloader = BackgroundDataLoader(config["background_folder"], seed=config["seed"])
    background = background_dataloader.get_image()
    
    if "Background" in config["transformations"].keys():
        for transformation in config["transformations"]["Background"]:
            background = transformation.apply(background)
    scene = Scene(background)
    scene.configure_positioning(config["positioning"])

    for object_label in config["foreground_objects"].keys():
        for i in range(config["object_counts"].get(object_label, 0)):
            image: ImgObject = dataloader.get_image(object_label)
            image.apply_transformations(config["transformations"][object_label])
        
            scene.add_foreground(image)
    
    scene.filters = config["filters"]
    scene.apply_filter()
    scene.configure_annotator(config["annotator"])
    scene.write("test.jpg", config["size"])
    scene.show(show_mask=False)
    bboxes = [fg.bbox.coordinates.astype(int) for fg in scene.foregrounds]
    coordinates = np.array([[bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]] for bbox in bboxes])
    return scene.background, coordinates, scene.annotator

def main(data_config_path: Path, transformation_config_path: Path):
    config = Config(".", transformation_config_path, data_config_path)
    generate_image(config)

def profile_main(data_config_path: str, transformation_config_path: str):
    pr = cProfile.Profile()
    pr.enable()
    
    main(data_config_path, transformation_config_path)
    
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())    
    
if __name__ == "__main__":
    #profile_main('data_config.yaml', 'transformation_config.yaml')
    main('/data/horse/ws/joka888b-syntheticImageGenerator/SyntheticImageGenerator/data_config.yaml', '/data/horse/ws/joka888b-syntheticImageGenerator/SyntheticImageGenerator/transformation_config.yaml')