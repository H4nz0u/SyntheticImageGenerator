import yaml
from transformations import Transformation
from filter import Filter
from object_position import BasePositionDeterminer
from image_management import ImageDataLoader
from utilities import create_transformation, create_positionDeterminer, create_filter, create_annotation, logger
from typing import List
from annotations import BaseAnnotator
import os

class Config:
    def __init__(self, base_path: str, transform_config_path: str, data_config_path: str):
        self.base_path = base_path
        self.transform_config = self.load_config(os.path.join(base_path, transform_config_path))
        self.data_config = self.load_config(os.path.join(base_path, data_config_path))
        self.config = {}
        self.merge_configs()
        try:
            self._validate_config()
        except Exception as e:
            logger.error(f'Error validating config: {e}')
            raise e

    def load_config(self, config_path: str):
        """Load a YAML configuration file and return a dictionary."""
        try:
            with open(config_path, 'r') as file:
                data = yaml.safe_load(file) or {}
                return data
        except yaml.YAMLError as exc:
            logger.error(f"Error loading YAML file {config_path}: {exc}")
            return {}
        except FileNotFoundError:
            logger.error(f"File not found: {config_path}")
            return {}

    def merge_configs(self):
        """Merge transformation settings into the data configuration for easy access."""
        self.config = {
            'foreground_objects': {k: os.path.join(self.data_config.get('root_path', ''), v) for k, v in self.data_config.get('foreground_objects', {}).items()},
            'transformations': self._parse_transformations(self.transform_config.get('transformations', {})),
            'filters' : self._parse_filters(self.transform_config.get('filters', [])),
            'blending_mode': self.transform_config.get('blending_mode', 'Standard'),
            'total_images': self.transform_config.get('total_images', 20),
            'seed': self.transform_config.get('seed', 42),
            'object_counts': self.transform_config.get('object_amount', {}),
            'root_path': self.data_config.get('root_path', ''),
            'background_folder': os.path.join(self.data_config.get('root_path', ''), self.data_config.get('background_folder', '')),
            'positioning': self._parse_positioning(self.transform_config.get('positioning', {})),
            'size': self.data_config.get('size', (800, 600)),
            'annotator': self._parse_annotation(self.data_config.get('annotation', None)),
        }     

    def _validate_config(self):
        self._validate_paths()
        self._validate_images()
        self._validate_transformations()
        self._validate_object_counts()
        
    def _validate_transformations(self):
        for label in self.config['transformations'].keys():
            if not label in self.config["foreground_objects"].keys() and label != "Background":
                raise ValueError(f'No path defined for label {label} in foreground_objects')
    
    def _validate_paths(self):
        if not self.config['root_path']:
            raise ValueError('Root path is not set')
        self._validate_path(self.config['root_path'])
        self._validate_path(self.config['background_folder'])
        for label, path in self.config['foreground_objects'].items():
            self._validate_path(path)
    
    def _validate_images(self):
        self._validate_image(self.config['background_folder'])
        for label, path in self.config['foreground_objects'].items():
            self._validate_image(path)
            
    def _validate_image(self, path: str):
        images = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.JPG')]
        if not images:
            raise ValueError(f'No images found in {path}')
    
    def _parse_positioning(self, positioning: dict) -> BasePositionDeterminer:
        return create_positionDeterminer(**positioning)
    
    def _parse_label(self, label: str) -> List[str]:
        if label == "all":
            return list(self.data_config['foreground_objects'].keys())
        return label.replace(" ", "").split(',')
    
    def _parse_transformation(self, label, transformations: dict, transforms: dict) -> Transformation:
        labels = self._parse_label(label)
        for l in labels:
            if l not in transforms.keys():
                transforms[l] = []
            for transformation in transformations:
                transforms[l].append(create_transformation(**transformation))
    
    def _parse_transformations(self, transformations: dict) -> List[Transformation]:
        transforms = {}
        for label in transformations.keys():
            self._parse_transformation(label, transformations[label], transforms)    
        return transforms
    
    
    def _parse_filters(self, filters: []) -> List[Filter]:
        fil = []
        for filter in filters:
            fil.append(create_filter(**filter))
        return fil
    
    def _parse_annotation(self, annotation: str) -> BaseAnnotator:
        return create_annotation(annotation)
        
    def _validate_path(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Path {path} does not exist')
    
    def _validate_object_counts(self):
        for label in self.config["foreground_objects"].keys():
            if not label in self.config["object_counts"].keys():
                raise ValueError(f'No object count defined for label {label} in object_counts')
            if not isinstance(self.config["object_counts"][label]["max"], int):
                raise ValueError(f'Object count for label {label} is not an integer! {self.config["object_counts"][label]}')
            if self.config["object_counts"][label]["max"] < 0:
                raise ValueError(f'Object count for label {label} is negative! {self.config["object_counts"][label]}')
            if self.config["object_counts"][label]["max"] == 0:
                raise ValueError(f'Object count for label {label} is zero! {self.config["object_counts"][label]}')
            if not isinstance(self.config["object_counts"][label]["prob"], float):
                raise ValueError(f'Object probability for label {label} is not an float! {self.config["object_counts"][label]}')
            if not 0 < self.config["object_counts"][label]["prob"] <= 1:
                raise ValueError(f'Object probability for label {label} is not between 0 and 1! {self.config["object_counts"][label]}')
        
    def save_config(self, config_path):
        with open(config_path, 'w') as stream:
            try:
                yaml.dump(self.params, stream)
            except yaml.YAMLError as exc:
                logger.error(exc)
    
    def get(self, key, default=None):
        return self.config.get(key, default)
                
    def __getitem__(self, key):
        if key not in self.config.keys():
            raise KeyError(f'Key {key} not found in config')
        return self.config[key]
    
    def __str__(self):
        return str(self.config)

if __name__ == "__main__":
    config = Config(".", "transformation_config.yaml", "data_config.yaml")
    print(config)