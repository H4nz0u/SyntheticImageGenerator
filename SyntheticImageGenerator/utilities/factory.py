from .registry import transformation_registry_dict
from .logging import logger
def create_transformation(name, **kwargs):
    transformation_cls = transformation_registry_dict.get(name)
    if transformation_cls is not None:
        logger.info(f"Creating transformation: {name}, {kwargs}")
        return transformation_cls(**kwargs)
    raise ValueError(f"No transformation registered with the name: {name}")

from .registry import positioning_registry_dict

def create_positionDeterminer(name, **kwargs):
    positionier_cls = positioning_registry_dict.get(name)
    if positionier_cls is not None:
        logger.info(f"Creating PositionDeterminer: {name}, {kwargs}")
        return positionier_cls(**kwargs)
    raise ValueError(f"No PositionDeterminer registered with the name: {name}")

from .registry import filter_registry_dict

def create_filter(name, **kwargs):
    filter_cls = filter_registry_dict.get(name)
    if filter_cls is not None:
        logger.info(f"Creating filter: {name}, {kwargs}")
        return filter_cls(**kwargs)
    raise ValueError(f"No filter registered with the name: {name}")

from .registry import annotation_registry_dict

def create_annotation(name, **kwargs):
    annotation_cls = annotation_registry_dict.get(name)
    if annotation_cls is not None:
        logger.info(f"Creating annotation: {name}, {kwargs}")
        return annotation_cls(**kwargs)
    raise ValueError(f"No annotation registered with the name: {name}")

from .registry import blending_registry_dict

def create_blending(name, **kwargs):
    blending_cls = blending_registry_dict.get(name)
    if blending_cls is not None:
        logger.info(f"Creating blending: {name}, {kwargs}")
        return blending_cls(**kwargs)
    raise ValueError(f"No blending registered with the name: {name}")