# Python modules
import os

# Installed pip modules
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

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
        Logger.warning('Id might change in future ucimlrepo versions\n')
    except Exception as e:
        Logger.error('Data could not be loaded, Exception: {}'.format(e))
        dataframe = None

    return dataframe


def preprocessing(data):
    '''Preprocessing a pandas health dataset.'''
    X_data = data.data.features
    y_data = data.data.targets
    # Check for full duplicates in the X_data
    try:
        assert len(X_data[X_data.duplicated()]) == 0
    except AssertionError:
        Logger.error('There are duplicates in the dataset which need to be handled first')
        raise ValueError

    flag_dict = {}
    pos_val_dict = {'age': list(range(0, 101)), 'sex': [0,1], 'cp': ['test']}
    pos_type_dict = {'age': int, 'sex': int}
    for i in X_data.columns:
        Logger.info('Checking feature "{}" for missing or wrong values and type'.format(i))
        assert X_data[i].isna().sum() == 0
        assert X_data[i].dtype == pos_type_dict[i]
        Logger.info('Type of feature is correct and there are no missing values')
        assert set(X_data[i].value_counts().sort_index().index.to_list()).issubset(pos_val_dict[i])
        Logger.info('Only valid values were used for the feature: {}\n'.format(i))
        if i == 'sex':
            break
    # Check age column if there is any missing value
    #Logger.info('Checking feature "age" for missing or wrong values and type')
    #Logger.info('Number of missing values {} and type of feature: {}'.format(X_data['age'].isna().sum(), X_data['age'].dtype))
    #Logger.info('Check for valid values of "age" {}\n'.format(X_data['age'].value_counts().sort_index().index.to_list()))

    # Check feature sex if there is any missing value
    #Logger.info('Checking feature "sex" for missing or wrong values and type')
    #Logger.info('Number of missing values {} and type of feature: {}'.format(X_data['sex'].isna().sum(), X_data['sex'].dtype))
    #Logger.info('Check for valid values of "sex" {}\n'.format(X_data['sex'].value_counts().sort_index().index.to_list()))

    # Check feature cp










if __name__ == '__main__':
    heart_data = load_data()
    #print(heart_data)
    preprocessing(heart_data)