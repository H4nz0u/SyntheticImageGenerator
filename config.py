import yaml

class Config:
    def __init__(self, config_path) -> None:
        self.params = self.load_config(config_path)
    
    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    
    def save_config(self, config_path):
        with open(config_path, 'w') as stream:
            try:
                yaml.dump(self.params, stream)
            except yaml.YAMLError as exc:
                print(exc)
                
    def __getitem__(self, key):
        if key not in self.params:
            raise KeyError(f'Key {key} not found in config')
        return self.params[key]