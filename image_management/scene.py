class Scene:
    def __init__(self, background, foregrounds) -> None:
        self.background = background
        self.foregrounds = foregrounds
        self.filters = []
    
    def add_filter(self, filter):
        self.filters.append(filter)
    
    def apply_filter(self):
        for filter in self.filters:
            filter.apply(self.background)
            
    def add_foreground(self, foreground):
        pass