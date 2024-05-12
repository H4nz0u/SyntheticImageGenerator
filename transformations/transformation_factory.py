from .transformation_registry import transformation_registry_dict

def create_transformation(name, **kwargs):
    transformation_cls = transformation_registry_dict.get(name)
    if transformation_cls is not None:
        print(f"Creating transformation: {name}, {kwargs}")
        return transformation_cls(**kwargs)
    raise ValueError(f"No transformation registered with the name: {name}")