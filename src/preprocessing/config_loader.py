import yaml
from yaml.loader import SafeLoader


class ConfigLoader:
    """
    This class is responsible for loading the configuration file.
    """

    @staticmethod
    def load_model_config(path):
        """
        TODO: Needs to be filled by Aykan
        """
        parameter = {}
        with open(path, "r", encoding="utf-8") as stream:
            try:
                # Converts yaml document to python object
                parameter = yaml.load(stream, Loader=SafeLoader)
            except yaml.YAMLError as e:
                print("CONFOG NOT FOUND")
                print(e)
                # TODO: Fehlerbehandlung einbauen, maybe default config???

        return parameter
