from pathlib import Path
from config import Config
from image_management.image import Image
from image_management.scene import Scene
from transformations.rotation import Rotate
def main(config_path: Path):
    config = Config(config_path)
    background = Image(config['background'])
    foreground = Image(config['foreground'])
    scene = Scene(background, [foreground])
    
    rotate = Rotate(config['angle'])
    

if __name__ == "__main__":
    main(Path('path/to/config.yml'))