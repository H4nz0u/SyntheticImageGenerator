filter_registry_dict = {}

def register_filter(cls: object):
    print("Registering filter class:", cls.__name__)
    filter_registry_dict[cls.__name__] = cls
    return cls
