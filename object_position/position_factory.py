from .position_registry import positioning_registry_dict

def create_positionDeterminer(name, **kwargs):
    positionier_cls = positioning_registry_dict.get(name)
    if positionier_cls is not None:
        print(f"Creating PositionDeterminer: {name}, {kwargs}")
        return positionier_cls(**kwargs)
    raise ValueError(f"No PositionDeterminer registered with the name: {name}")