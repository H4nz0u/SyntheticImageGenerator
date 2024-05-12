from .filter_registry import filter_registry_dict

def create_filter(name, **kwargs):
    filter_cls = filter_registry_dict.get(name)
    if filter_cls is not None:
        return filter_cls(**kwargs)
    raise ValueError(f"No filter registered with the name: {name}")