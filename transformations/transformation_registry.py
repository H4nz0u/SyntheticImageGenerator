from functools import wraps
transformation_registry_dict = {}

def register_transformation(cls: object):
    print("Registering transformation class:", cls.__name__)
    transformation_registry_dict[cls.__name__] = cls
    return cls
