# Python modules
import os

# Installed pip modules
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

# Own created modules
from utils import MakeLogger
from utils import YamlHandler
from utils import Visualizer
from preprocessing import preprocess, visualize
#from utils import memorizer

# Loggers
prep_log = MakeLogger().costum_log(filename='preprocess.log')
feature_log = MakeLogger().costum_log(filename='features.log')

# Yaml file handler
yhadl = YamlHandler()

# Paths
settings_path = os.path.join(os.path.dirname(os.getcwd()), 'settings')
fig_path = os.path.join(os.path.dirname(os.getcwd()), 'figures')
p_file_path = os.path.join(settings_path, 'parameter.yml')
d_path = os.path.join(os.path.dirname(os.getcwd()), 'data')
yml_obj = yhadl.loader(p_file_path)


def load_data(logObj):
    """This function fetches data directly from the module ucimlrepo. Therefore there is no
    need to import the data manually from the /data folder
    return: pandas dataframe"""
    try:
        dataframe = fetch_ucirepo(id=45)
        logObj.info('Function call load_data was successfull')
        logObj.debug('Heart disease data has been loaded')
        logObj.warning('Id might change in future ucimlrepo versions\n')
    except Exception as e:
        logObj.error('Data could not be loaded, Exception: {}'.format(e))
        dataframe = None

    return dataframe


def main():
    heart_disease_data = load_data(prep_log)
    X_data, y_data = preprocess(heart_disease_data, prep_log, feature_log, yml_obj)
    visualize(Visualizer(X_data, y_data), fig_path, prep_log)

    # Save preprocessed Data
    X_data.to_pickle(os.path.join(d_path, 'X_data.pkl'))
    y_data.to_pickle(os.path.join(d_path, 'y_data.pkl'))


if __name__ == '__main__':
    main()

