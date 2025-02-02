import os
from ucimlrepo import fetch_ucirepo
import pandas as pd

from utils import MakeLogger
from utils import YamlHandler
from utils import Visualizer
from preprocessing import preprocess, visualize
from machine_learning import MachineLearning

# Loggers
prep_log = MakeLogger().costum_log(filename='preprocess.log')
feature_log = MakeLogger().costum_log(filename='features.log')
ml_log = MakeLogger().costum_log(filename='learning.log')

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


def prepare():
    """ Prepares the data from the ucirepo by performing multiple pre-processing steps.
    To skip this step in Future runs of the Skript the processed data is pickled and saved."""
    heart_disease_data = load_data(prep_log)
    X_data, y_data = preprocess(heart_disease_data, prep_log, feature_log, yml_obj)
    visualize(Visualizer(X_data, y_data), fig_path, prep_log)

    # Save preprocessed Data
    X_data.to_pickle(os.path.join(d_path, 'X_data.pkl'))
    y_data.to_pickle(os.path.join(d_path, 'y_data.pkl'))


def learn():
    """ Applying machine learning algorithms to the data. Currently working with 3 different algorithms
    which will be updated to 5 in the future."""
    X_data = pd.read_pickle(os.path.join(d_path, 'X_data.pkl'))
    y_data = pd.read_pickle(os.path.join(d_path, 'y_data.pkl'))
    ML_obj = MachineLearning(X_data, y_data, yml_obj)

    # Logistic Regression
    log_reg_path = os.path.join(os.getcwd(), 'Results/Logistic_Regression')
    log_reg_acc, log_reg_cr = ML_obj.log_reg(log_reg_path)
    ml_log.info('The accuracy of the Logistic Regression model after cross validation is: {}'.format(log_reg_acc))
    ml_log.info('This is the full classification report of the model \n\n{}\n'.format(log_reg_cr))

    # k-Nearest Neighbors
    k_nearest_path = os.path.join(os.getcwd(), 'Results/KNearest')
    grid_model, knn_acc, knn_cr = ML_obj.k_nearest(k_nearest_path)
    ml_log.info('The accuracy of the k-Nearest model after hyperparameter tuning is: {}'.format(knn_acc))
    ml_log.info('These are the best parameters from training {}'.format(grid_model.best_params_))
    ml_log.info('This is the full classification report of the model \n\n{}\n'.format(knn_cr))

    # RandomForest
    rf_path = os.path.join(os.getcwd(), 'Results/RandomForest')
    grid_model, rf_acc, rf_cr = ML_obj.random_forest(rf_path)
    ml_log.info('The accuracy of the Randomforest model after hyperparameter tuning is: {}'.format(rf_acc))
    ml_log.info('These are the best parameters from training {}'.format(grid_model.best_params_))
    ml_log.info('This is the full classification report of the model \n\n{}\n'.format(rf_cr))

    # Gradient boosted Trees
    # Upcoming

    # Voting classifier (Decision Tree, Logistic Regression, KNN)
    # Upcoming

if __name__ == '__main__':
    prepare()
    learn()

