from pathlib import Path
from .config import Config
from .image_management import ImageDataLoader, Scene, BackgroundDataLoader, ImgObject
import cv2
import numpy as np
from .utilities import logger, get_cached_dataframe
from .utilities.dataframe_cache import dataframe_cache
import faulthandler
faulthandler.enable()
import cProfile
import pstats
import io
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import argparse
from typing import Tuple
from .annotations import BaseAnnotator

def generate_image(config: Config, background_dataloader: BackgroundDataLoader, dataloader: ImageDataLoader) -> Scene:
    background = background_dataloader.get_image()
    #background = cv2.resize(background, (config["size"][0], config["size"][1]))
    
    if "Background" in config["transformations"].keys():
        for transformation in config["transformations"]["Background"]:
            background = transformation.apply(background)
    
    for transformation_cls in config["transformations"]:
        for transformation in config["transformations"][transformation_cls]:
            if transformation.__class__.__name__ in ["ScaleToArea", "ScaleToAreaFromDataFrame", "RandomScaleToArea"]:
                transformation.background_size = background.shape[0] * background.shape[1]
                
    
    scene = Scene(background)
    scene.configure_annotator(config["annotator"])
    scene.configure_positioning(config["positioning"])
    scene.configure_blending(config["blending"])

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
    scene.show(show_mask=False, show_bbox=True, show_class=False, show_segmentation=False)
    #scene.write(Path("output.jpg"), config["size"])
    
    for df_cache in dataframe_cache.values():
        df_cache.reset()
    
    return scene

def generate_image_wrapper(config: Config) -> Tuple[cv2.typing.MatLike, np.ndarray, BaseAnnotator]:
    dataloader = ImageDataLoader(".",  config["foreground_objects"], seed=config["seed"])
    background_dataloader = BackgroundDataLoader(config["background_folder"], seed=config["seed"])
    scene = generate_image(config, background_dataloader, dataloader)
    bboxes = [fg.bbox.coordinates.astype(int) for fg in scene.foregrounds]
    coordinates = np.array([[bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]] for bbox in bboxes])
    return scene.background, coordinates, scene.annotator

def main(data_config_path: Path, transformation_config_path: Path, output_path: Path):
    config = Config(Path("."), transformation_config_path, data_config_path)
    print(config)
    print(dataframe_cache)
    with logging_redirect_tqdm():
        dataloader = ImageDataLoader(".",  config["foreground_objects"], seed=config["seed"])
        background_dataloader = BackgroundDataLoader(config["background_folder"], seed=config["seed"])
        for i in tqdm(range(config["total_images"])):
            scene = generate_image(config, background_dataloader, dataloader)
            height, width, _ = scene.background.shape
            scene.write(output_path / f"image_{i}.jpg", (width, height))

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
    parser.add_argument('--data-config', type=Path, required=False, help="Path to the data config YAML file", default='/data/horse/ws/joka888b-syntheticImageGenerator/SyntheticImageGenerator/SyntheticImageGenerator/data_config.yaml')
    parser.add_argument('--transformation-config', type=Path, required=False, help="Path to the transformation config YAML file", default='/data/horse/ws/joka888b-syntheticImageGenerator/SyntheticImageGenerator/SyntheticImageGenerator/transformation_config.yaml')
    parser.add_argument('--output-path', type=Path, required=False, help="Path to the output directory", default='/data/horse/ws/joka888b-syntheticImageGenerator/SyntheticImageGenerator/output')
    parser.add_argument('--profile', action='store_true', help="Profile the code")
    args = parser.parse_args()

    if args.profile:
        profile_main(args.data_config, args.transformation_config, args.output_path)
    else:
        main(args.data_config, args.transformation_config, args.output_path)