import yaml
from yaml.loader import SafeLoader


class ConfigService:
    def __init__(self):
        pass

    def loadModelConfig(self, path):
        # opening a file
        parameter = {}
        with open(path, 'r') as stream:
            try:
                # Converts yaml document to python object
                parameter = yaml.load(stream, Loader=SafeLoader)
            except yaml.YAMLError as e:
                print('CONFOG NOT FOUND')
                print(e)
                # todo Fehlerbehandlung einbauen, maybe defualt config???

        return parameter
