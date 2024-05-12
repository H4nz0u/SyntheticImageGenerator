import yaml
from transformations import create_transformation, Transformation
from image_management import ImageDataLoader
from typing import List
import os

class Config:
    def __init__(self, base_path: str, transform_config_path: str, data_config_path: str):
        self.base_path = base_path
        self.transform_config = self.load_config(os.path.join(base_path, transform_config_path))
        self.data_config = self.load_config(os.path.join(base_path, data_config_path))
        self.config = {}
        self.merge_configs()

    def load_config(self, config_path: str):
        """Load a YAML configuration file and return a dictionary."""
        try:
            with open(config_path, 'r') as file:
                data = yaml.safe_load(file) or {}
                return data
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file {config_path}: {exc}")
            return {}
        except FileNotFoundError:
            print(f"File not found: {config_path}")
            return {}

    def merge_configs(self):
        """Merge transformation settings into the data configuration for easy access."""
        self.config = {
            'transformations': self._parse_transformations(self.transform_config.get('transformations', {})),
            'filters': self.transform_config.get('filters', {}),
            'blending_mode': self.transform_config.get('blending_mode', 'Standard'),
            'total_images': self.transform_config.get('total_images', 20),
            'seed': self.transform_config.get('seed', 42),
            'object_counts': self.transform_config.get('object_amount', {}),
            'root_path': self.data_config.get('root_path', ''),
            'background_folder': os.path.join(self.data_config.get('root_path', ''), self.data_config.get('background_folder', '')),
            'foreground_objects': {k: os.path.join(self.data_config.get('root_path', ''), v) for k, v in self.data_config.get('foreground_objects', {}).items()}
        }     

    def _parse_transformations(self, transformations: dict) -> List[Transformation]:
        transforms = {}
        for label in transformations.keys():
            if label not in transforms.keys():
                transforms[label] = []
            for transformation in transformations[label]:
                transforms[label].append(create_transformation(**transformation))
            return transforms
        
    def save_config(self, config_path):
        with open(config_path, 'w') as stream:
            try:
                yaml.dump(self.params, stream)
            except yaml.YAMLError as exc:
                print(exc)
    
    def get(self, key, default=None):
        return self.config.get(key, default)
                
    def __getitem__(self, key):
        if key not in self.params:
            raise KeyError(f'Key {key} not found in config')
        return self.params[key]
    
    def __str__(self):
        return str(self.config)

if __name__ == "__main__":
    config = Config(".", "transformation_config.yaml", "data_config.yaml")
    print(config)