# Python modules
import os

# Installed pip modules
from ucimlrepo import fetch_ucirepo

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


def load_data():
    '''This function fetches data directly from the module ucimlrepo. Therefore there is no
    need to import the data manually from the /data folder
    return: pandas dataframe'''
    try:
        dataframe = fetch_ucirepo(id=45)
        Logger.info('Function call load_data was successfull')
        Logger.debug('Heart disease data has been loaded')
        Logger.warning('Id might change in future ucimlrepo versions')
    except:
        Logger.error('Heart disease data could not be loaded')

    return dataframe












if __name__ == '__main__':
    df = load_data()