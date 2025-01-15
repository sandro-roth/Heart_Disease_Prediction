import yaml

class YamlHandler:
    def loader(self, filepath):
        '''loading a yamlfile
        filepath: str, path to .yml'''
        with open(filepath, 'r') as file:
            return yaml.safe_load(file)

    def dumper(self, data, filepath):
        '''dumping dictionary to yaml file.
        data: dict, data to dump to a file
        filepath: str, path to file'''
        with open(filepath, 'w') as file:
            return yaml.safe_dump(data, file)