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
#from utils import memorizer

# Initializing Project by setting up logger and parameter settings
prep_log = MakeLogger().costum_log(filename='preprocess.log')
feature_log = MakeLogger().costum_log(filename='features.log')
yhadl = YamlHandler()

# paths
settings_path = os.path.join(os.path.dirname(os.getcwd()), 'settings')
fig_path = os.path.join(os.path.dirname(os.getcwd()), 'figures')
p_file_path = os.path.join(settings_path, 'parameter.yml')
data_path = os.path.join(os.path.dirname(os.getcwd()), 'data')

parameter = yhadl.loader(p_file_path)


def load_data():
    """This function fetches data directly from the module ucimlrepo. Therefore there is no
    need to import the data manually from the /data folder
    return: pandas dataframe"""
    try:
        dataframe = fetch_ucirepo(id=45)
        prep_log.info('Function call load_data was successfull')
        prep_log.debug('Heart disease data has been loaded')
        prep_log.warning('Id might change in future ucimlrepo versions\n')
    except Exception as e:
        prep_log.error('Data could not be loaded, Exception: {}'.format(e))
        dataframe = None

    return dataframe


def preprocessing(data):
    """Preprocessing a pandas health dataset."""
    X_data = data.data.features
    y_data = data.data.targets
    # Check for full duplicates in the X_data
    try:
        assert len(X_data[X_data.duplicated()]) == 0
        prep_log.info('There are no full duplicates in the dataset!\n')
    except AssertionError:
        prep_log.error('There are duplicates in the dataset which need to be handled first')
        raise ValueError

    flags = set()
    f_val_d = parameter['preprocessing']['f_val_d']
    f_type_d = parameter['preprocessing']['f_type_d']

    # Remove SettingWithCopyWarnings for the dtype part setting
    pd.options.mode.chained_assignment = None

    for i in X_data.columns:
        try:
            # Check for missing values in the current column
            prep_log.info('Checking feature "{}" for missing or wrong values and type'.format(i))
            if X_data[i].isna().sum() != 0:
                raise ValueError

            # Check if the values are in the correct range
            if X_data[i].between(*f_val_d[i]).all() != 1:
                raise Exception('The Value is not in the right range')
            prep_log.info('Only valid values were used for the feature: {}'.format(i))

            # reassure that the type is set correctly
            X_data[i] = X_data[i].astype(f_type_d[i])
            if X_data[i].dtype != f_type_d[i]:
                raise TypeError
            prep_log.info('Type of feature: "{}" is also set correctly as : "{}"\n'.format(i, X_data[i].dtype))

        except ValueError:
            prep_log.info('There are missing values in the feature "{}"'.format(i))
            prep_log.warning('The feature "{}" will be handled later on and is currently stored\n'.format(i))
            flags.add(i)

        except TypeError:
            prep_log.info('Type warning appeared for "{}" which means the data source was changed'.format(i))
            prep_log.warning('The type is not handled further and may lead to complications later on\n')

        except Exception as error:
            prep_log.info('Error-message: {}'.format(repr(error)))
            prep_log.warning('Data source was changed and will influence results of ML Algorithm applied later on\n')

    # Checking columns with missing values
    for i in flags:
        if i == 'ca':
            feature_log.debug('This feature "ca" describes number of major vessels (0-3) and therefore cannot be estimated')
            feature_log.info('The number of missing values is: {}'.format(X_data[i].isna().sum()))
            feature_log.warning('Since this is less than 5%. The indices of the missing values are deleted from the data')
            drop_list = X_data[X_data['ca'].isna()].index.to_list()
            feature_log.warning('The indices {} of X_data and y_data are dropped\n'.format(drop_list))
            X_data.drop(labels=drop_list, axis=0, inplace=True)
            y_data.drop(labels=drop_list, axis=0, inplace=True)

        elif i == 'thal':
            pass
            feature_log.debug('This feature "thal" is the inherited blood disorder of no producing enough hemoglobin it cannot be estimated')
            feature_log.info('The number of missing values is: {}'.format(X_data[i].isna().sum()))
            feature_log.warning('Since this is less than 5%. The indices of the missing values are deleted from the data')

            drop_list = X_data[X_data['thal'].isna()].index.to_list()
            feature_log.warning('The indices {} of X_data and y_data are dropped\n'.format(drop_list))
            X_data.drop(labels=drop_list, axis=0, inplace=True)
            y_data.drop(labels=drop_list, axis=0, inplace=True)

        else:
            feature_log.debug('There is an unexpected feature "{}"'.format(i))
            feature_log.critical('This unexpected feature most likely breaks the code further down')


    # Reset the indices of X and y data
    X_data.reset_index(inplace=True, drop=True)
    y_data.reset_index(inplace=True, drop=True)

    # Change target values to present or absence of heart disease
    y_data.loc[y_data['num'] > 0] = 1
    y_data.rename(columns={'num': 'target'}, inplace=True)

    # Set the SettingWithCopyWarnings to "warn" again
    pd.options.mode.chained_assignment = 'warn'

    # Visualize Data as EDA to look for outliers
    eda = Visualizer(X_data, y_data)
    eda.pairplot()
    eda.save_pic(fig_path)
    eda.correlation()
    eda.save_pic(fig_path)
    eda.barplot('sex')
    eda.save_pic(fig_path)
    eda.barplot('fbs')
    eda.save_pic(fig_path)
    eda.barplot('exang')
    eda.save_pic(fig_path)
    eda.barplot('cp')
    eda.save_pic(fig_path)
    eda.boxplot()
    eda.save_pic(fig_path)

    prep_log.info('The features will not be further pre-processed.')
    prep_log.info('Dataset will be stored in directory "data".')
    prep_log.warning('Outliers in some features may affect performance of model. This has to be checked later on.')

    X_data.to_pickle(os.path.join(data_path, 'X_data.pkl'))
    y_data.to_pickle(os.path.join(data_path, 'y_data.pkl'))


if __name__ == '__main__':
    heart_data = load_data()
    preprocessing(heart_data)