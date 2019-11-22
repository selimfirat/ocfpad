
class Data:


    def __init__(self, name):
        self.name = name
        self.load_data(name)

    def load_data(self, name):
        data_func = getattr(self, name)

        data_func()
