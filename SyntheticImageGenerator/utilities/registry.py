from functools import wraps
from .logging import logger
transformation_registry_dict = {}

def register_transformation(cls):
    logger.info(f"Registering transformation class: {cls.__name__}")
    transformation_registry_dict[cls.__name__] = cls
    return cls

positioning_registry_dict = {}

def register_positionDeterminer(cls):
    logger.info(f"Registering PositionDeterminer class: {cls.__name__}")
    positioning_registry_dict[cls.__name__] = cls
    return cls

filter_registry_dict = {}

def register_filter(cls):
    logger.info(f"Registering filter class: {cls.__name__}")
    filter_registry_dict[cls.__name__] = cls
    return cls

annotation_registry_dict = {}

def register_annotation(cls):
    logger.info(f"Registering annotation class: {cls.__name__}")
    annotation_registry_dict[cls.__name__] = cls
    return cls

blending_registry_dict = {}

def register_blending(cls):
    logger.info(f"Registering blending class: {cls.__name__}")
    blending_registry_dict[cls.__name__] = cls
    return cls