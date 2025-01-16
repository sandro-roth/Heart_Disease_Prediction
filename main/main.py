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
        Logger.info('There are no full duplicates in the dataset!\n')
    except AssertionError:
        Logger.error('There are duplicates in the dataset which need to be handled first')
        raise ValueError


    print(X_data['oldpeak'].min())
    print(X_data['oldpeak'].max())
    print(X_data['oldpeak'].value_counts().sort_index())
    print(X_data['oldpeak'].dtype)
    flag_dict = {}
    f_val_d = parameter['preprocessing']['f_val_d']
    f_type_d = parameter['preprocessing']['f_type_d']
    # Remove SettingWithCopyWarnings for the dtype part setting
    pd.options.mode.chained_assignment = None
    for i in X_data.columns:
        try:
            Logger.info('Checking feature "{}" for missing or wrong values and type'.format(i))
            assert X_data[i].isna().sum() == 0
            assert X_data[i].between(*f_val_d[i]).all() == 1
            Logger.info('Only valid values were used for the feature: {}'.format(i))
            X_data[i] = X_data[i].astype(f_type_d[i])
            assert X_data[i].dtype == f_type_d[i]
            Logger.info('Type of feature: "{}" is also set correctly as: "{}"\n'.format(i, X_data[i].dtype))


            if i == 'oldpeak':
                break

        except AssertionError:
            print('Let me see which error it is!')
            print(X_data[i].dtype)

    pd.options.mode.chained_assignment = 'warn'
    print('---------------\n', X_data['oldpeak'].dtype) # --> Delete!











if __name__ == '__main__':
    heart_data = load_data()
    preprocessing(heart_data)
    #pre_para = parameter['preprocessing']['pos_val_dict']
    #print(pre_para['age'])
    #print(pre_para['sex'])
