# Python modules
import os

# Installed pip modules


# Own created modules
from utils import MakeLogger
from utils import YamlHandler
from utils import memorizer

# Initializing Project by setting up logger and parameter settings
Logger = MakeLogger().costum_log(filename='main.log')
yhadl = YamlHandler()
settings_path = os.path.join(os.path.dirname(os.getcwd()), 'settings')
p_file_path = os.path.join(settings_path, 'parameter.yml')
parameter = yhadl.loader(p_file_path)













if __name__ == '__main__':
    pass