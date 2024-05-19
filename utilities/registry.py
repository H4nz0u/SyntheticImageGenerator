from functools import wraps
from .logging import logger
transformation_registry_dict = {}

def register_transformation(cls: object):
    logger.info(f"Registering transformation class: {cls.__name__}")
    transformation_registry_dict[cls.__name__] = cls
    return cls

positioning_registry_dict = {}

def register_positionDeterminer(cls: object):
    logger.info(f"Registering PositionDeterminer class: {cls.__name__}")
    positioning_registry_dict[cls.__name__] = cls
    return cls

filter_registry_dict = {}

def register_filter(cls: object):
    logger.info(f"Registering filter class: {cls.__name__}")
    filter_registry_dict[cls.__name__] = cls
    return cls

annotation_registry_dict = {}

def register_annotation(cls: object):
    logger.info(f"Registering annotation class: {cls.__name__}")
    annotation_registry_dict[cls.__name__] = cls
    return cls