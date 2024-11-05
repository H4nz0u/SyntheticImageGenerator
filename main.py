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
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import argparse
def generate_image(config: Config):
    dataloader = ImageDataLoader(".",  config["foreground_objects"], seed=config["seed"])
    background_dataloader = BackgroundDataLoader(config["background_folder"], seed=config["seed"])
    background = background_dataloader.get_image()
    background = cv2.resize(background, (config["size"][0], config["size"][1]))
    
    if "Background" in config["transformations"].keys():
        for transformation in config["transformations"]["Background"]:
            background = transformation.apply(background)
    scene = Scene(background)
    scene.configure_annotator(config["annotator"])
    scene.configure_positioning(config["positioning"])

    for object_label in config["foreground_objects"].keys():
        for i in range(config["object_counts"].get(object_label, 0)):
            image: ImgObject = dataloader.get_image(object_label)
            if scene.annotator.overwrite_classes and image.cls in scene.annotator.overwrite_classes.values():
                new_class = None
                for key, value in scene.annotator.overwrite_classes.items():
                    if value == image.cls:
                        new_class = key
                        break
                if new_class:
                    image.cls = new_class
                else:
                    logger.error(f"Class {image.cls} not found in overwrite_classes")
            image.apply_transformations(config["transformations"][object_label])
        
            scene.add_foreground(image)
    
    scene.filters = config["filters"]
    scene.apply_filter()
    #scene.show()
    #scene.write(Path("output.jpg"), config["size"])
    return scene

def generate_image_wrapper(config: Config):
    scene = generate_image(config)
    bboxes = [fg.bbox.coordinates.astype(int) for fg in scene.foregrounds]
    coordinates = np.array([[bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]] for bbox in bboxes])
    return scene.background, coordinates, scene.annotator

def main(data_config_path: Path, transformation_config_path: Path, output_path: Path):
    config = Config(Path("."), transformation_config_path, data_config_path)
    with logging_redirect_tqdm():
        for i in tqdm(range(config["total_images"])):
            scene = generate_image(config)
            scene.write(output_path / f"image_{i}.jpg", config["size"])

def profile_main(data_config_path: Path, transformation_config_path: Path, output_path: Path):
    pr = cProfile.Profile()
    pr.enable()
    
    main(data_config_path, transformation_config_path, output_path)
    
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Image Generator")
    parser.add_argument('--data-config', type=Path, required=False, help="Path to the data config YAML file", default='/data/horse/ws/joka888b-syntheticImageGenerator/SyntheticImageGenerator/data_config.yaml')
    parser.add_argument('--transformation-config', type=Path, required=False, help="Path to the transformation config YAML file", default='/data/horse/ws/joka888b-syntheticImageGenerator/SyntheticImageGenerator/transformation_config.yaml')
    parser.add_argument('--output-path', type=Path, required=False, help="Path to the output directory", default='/data/horse/ws/joka888b-syntheticImageGenerator/SyntheticImageGenerator/output')
    parser.add_argument('--profile', action='store_true', help="Profile the code")
    args = parser.parse_args()

    if args.profile:
        profile_main(args.data_config, args.transformation_config, args.output_path)
    else:
        main(args.data_config, args.transformation_config, args.output_path)