from pathlib import Path
from config import Config
from image_management.image import Image
from image_management.scene import Scene
from transformations import create_transformation, transformation_registry_dict

config = {
    "transformations": [
        {"name": "Rotate", "angle": 90},
        {"name": "Scale", "factor": 1.5}
    ]
}

def main(config_path: Path):
    transforms = []
    for transformations in config["transformations"]:
        transforms.append(create_transformation(**transformations))
    print(transforms)
    
    
if __name__ == "__main__":
    print(transformation_registry_dict)
    main(Path('path/to/config.yml'))