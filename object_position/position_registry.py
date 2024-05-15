positioning_registry_dict = {}

def register_positionDeterminer(cls: object):
    print("Registering PositionDeterminer class:", cls.__name__)
    positioning_registry_dict[cls.__name__] = cls
    return cls
