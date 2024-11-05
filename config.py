import yaml
from transformations import Transformation
from filter import Filter
from object_position import BasePositionDeterminer
from image_management import ImageDataLoader
from utilities import create_transformation, create_positionDeterminer, create_filter, create_annotation, logger, get_cached_dataframe
from typing import List, Dict
from annotations import BaseAnnotator
from pathlib import Path
import os

class Config:
    data_config: Dict = {}
    def __init__(self, 
                 base_path: Path, 
                 transform_config_path: Path, 
                 data_config_path: Path):
        self.base_path = base_path
        self.transform_config = self.load_config(os.path.join(base_path, transform_config_path))
        self.data_config = self.load_config(os.path.join(base_path, data_config_path))
        self.config = {}
        self._load_dataframes()
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
            'size': self.data_config.get('size', (2304, 1728)),
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
    
    def _parse_transformation(self, label, transformations: dict, transforms: dict) -> None:
        labels = self._parse_label(label)
        for label in labels:
            if label not in transforms.keys():
                transforms[label] = []
            for transformation in transformations:
                transforms[label].append(create_transformation(**transformation))
    
    def _parse_transformations(self, transformations: dict) -> Dict[str, List[Transformation]]:
        transforms: Dict[str, List[Transformation]] = {}
        for label in transformations.keys():
            self._parse_transformation(label, transformations[label], transforms)    
        return transforms
    
    
    def _parse_filters(self, filters: List = []) -> List[Filter]:
        fil = []
        for filter in filters:
            fil.append(create_filter(**filter))
        return fil
    
    def _parse_annotation(self, annotation: str) -> BaseAnnotator:
        if self.data_config is not None:
            overwrite_classes = self.data_config.get('overwrite_classes', {})
        else:
            overwrite_classes = {}
        annotator = create_annotation(annotation, overwrite_classes=overwrite_classes)
        return annotator

        
    def _validate_path(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Path {path} does not exist')
    
    def _validate_object_counts(self):
        for label in self.config["foreground_objects"].keys():
            if not label in self.config["object_counts"].keys():
                raise ValueError(f'No object count defined for label {label} in object_counts')
            if not isinstance(self.config["object_counts"][label], int):
                raise ValueError(f'Object count for label {label} is not an integer')
            if self.config["object_counts"][label] < 0:
                raise ValueError(f'Object count for label {label} is negative')
            
    def _load_dataframes(self):
        self.dataframes = {}
        for name, path in self.data_config.get("dataframes", {}).items():
            get_cached_dataframe(name, path)    
    
    def get(self, key, default=None):
        return self.config.get(key, default)
                
    def __getitem__(self, key):
        if key not in self.config.keys():
            raise KeyError(f'Key {key} not found in config')
        return self.config[key]
    
    def set_transformations(self, transformations: Dict[str, List[float]]) -> None:
        self.config['transformations'] = self._parse_transformations(transformations)

    def set_filters(self, filters: List[Dict[str, float]]) -> None:
        self.config['filters'] = self._parse_filters(filters)

    def set_blending_mode(self, blending_mode: str) -> None:
        self.config['blending_mode'] = blending_mode

    def set_positioning(self, positioning: Dict) -> None:
        self.config['positioning'] = create_positionDeterminer(**positioning)

    def set_foreground_objects(self, foreground_objects: Dict[str, str]) -> None:
        self.config['foreground_objects'] = {
            k: os.path.join(self.config['root_path'], v) for k, v in foreground_objects.items()
        }

    def set_background_folder(self, background_folder: str) -> None:
        self.config['background_folder'] = os.path.join(self.config['root_path'], background_folder)

    def set_root_path(self, root_path: str) -> None:
        self.config['root_path'] = root_path

    def set_object_counts(self, object_counts: Dict[str, int]) -> None:
        self.config['object_counts'] = object_counts

    def _convert_to_dict(self):
        def serialize(obj):
            # This helper function filters out non-serializable attributes
            if isinstance(obj, Transformation) or isinstance(obj, Filter):
                # Only include JSON serializable properties
                return {k: v for k, v in obj.__dict__.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
            return obj
        config_dict = {
            'transformations': {label: [serialize(t) for t in trans] for label, trans in self.config['transformations'].items()},            'filters': [f.__dict__ for f in self.config['filters']],
            'filters': [serialize(f) for f in self.config['filters']],
            'blending': self.config['blending_mode'],
            'positioning': self.config['positioning'].__dict__ if self.config['positioning'] else None,
            'total_images': self.config['total_images'],
            'seed': self.config['seed'],
            'object_amount': self.config['object_counts'],
            'foreground_objects': self.config['foreground_objects'],
            'background_folder': self.config['background_folder'],
            'root_path': self.config['root_path'],
            'size': self.config['size'],
            'annotation': self.config['annotator'].__class__.__name__ if self.config['annotator'] else None
        }
        return config_dict

    def __str__(self):
        return str(self.config)
    
class OptimizationConfig(Config):
    def __init__(self):
        self.config = {
            'transformations': {},
            'filters': [],
            'blending_mode': 'Standard',
            'positioning': None,
            'foreground_objects': {
                "Typlabel-China": r"/data/horse/ws/joka888b-syntheticImageGenerator/SyntheticImageGenerator/images/ISI-Typlabelchina-Schilder/"
            },
            'background_folder': r'/data/horse/ws/joka888b-syntheticImageGenerator/SyntheticImageGenerator/images/cars/',
            'root_path': r'/data/horse/ws/joka888b-syntheticImageGenerator/SyntheticImageGenerator/images/',
            'object_counts': {"Typlabel-China": 1},
            'seed': 42,
            'total_images': 1,
            'size': (1300, 867),
            'annotator': self._parse_annotation("PascalVOC")
        }
        self.data_config = {}


if __name__ == "__main__":
    config = Config(Path("."), Path("transformation_config.yaml"), Path("data_config.yaml"))
    print(config)