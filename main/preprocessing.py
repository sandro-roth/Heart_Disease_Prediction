import pandas as pd


def preprocess(data, p_log_obj, f_log_obj, yml_obj):
    """
    Preprocessing a pandas health dataset.
    :param data: dataframe, includes features and target data
    :param p_log_obj: obj, Instance of Logger class for preprocessing
    :param f_log_obj: obj, Instance of Logger class for features
    :param yml_obj: obj, Instance of yaml-handler to load parameter settings
    :return: X_data and y_data, preprocessed dataset
    """
    X_data = data.data.features
    y_data = data.data.targets
    # Check for full duplicates in the X_data
    try:
        assert len(X_data[X_data.duplicated()]) == 0
        p_log_obj.info('There are no full duplicates in the dataset!\n')
    except AssertionError:
        p_log_obj.error('There are duplicates in the dataset which need to be handled first')
        raise ValueError

    flags = set()
    f_val_d = yml_obj['preprocessing']['f_val_d']
    f_type_d = yml_obj['preprocessing']['f_type_d']

    # Remove SettingWithCopyWarnings for the dtype part setting
    pd.options.mode.chained_assignment = None

    for i in X_data.columns:
        try:
            # Check for missing values in the current column
            p_log_obj.info('Checking feature "{}" for missing or wrong values and type'.format(i))
            if X_data[i].isna().sum() != 0:
                raise ValueError

            # Check if the values are in the correct range
            if X_data[i].between(*f_val_d[i]).all() != 1:
                raise Exception('The Value is not in the right range')
            p_log_obj.info('Only valid values were used for the feature: {}'.format(i))

            # reassure that the type is set correctly
            X_data[i] = X_data[i].astype(f_type_d[i])
            if X_data[i].dtype != f_type_d[i]:
                raise TypeError
            p_log_obj.info('Type of feature: "{}" is also set correctly as : "{}"\n'.format(i, X_data[i].dtype))

        except ValueError:
            p_log_obj.info('There are missing values in the feature "{}"'.format(i))
            p_log_obj.warning('The feature "{}" will be handled later on and is currently stored\n'.format(i))
            flags.add(i)

        except TypeError:
            p_log_obj.info('Type warning appeared for "{}" which means the data source was changed'.format(i))
            p_log_obj.warning('The type is not handled further and may lead to complications later on\n')

        except Exception as error:
            p_log_obj.info('Error-message: {}'.format(repr(error)))
            p_log_obj.warning('Data source was changed and will influence results of ML Algorithm applied later on\n')

    # Checking columns with missing values
    for i in flags:
        if i == 'ca':
            f_log_obj.debug('This feature "ca" describes number of major vessels (0-3) and therefore cannot be estimated')
            f_log_obj.info('The number of missing values is: {}'.format(X_data[i].isna().sum()))
            f_log_obj.warning('Since this is less than 5%. The indices of the missing values are deleted from the data')
            drop_list = X_data[X_data['ca'].isna()].index.to_list()
            f_log_obj.warning('The indices {} of X_data and y_data are dropped\n'.format(drop_list))
            X_data.drop(labels=drop_list, axis=0, inplace=True)
            y_data.drop(labels=drop_list, axis=0, inplace=True)

        elif i == 'thal':
            pass
            f_log_obj.debug('This feature "thal" is the inherited blood disorder of no producing enough hemoglobin it cannot be estimated')
            f_log_obj.info('The number of missing values is: {}'.format(X_data[i].isna().sum()))
            f_log_obj.warning('Since this is less than 5%. The indices of the missing values are deleted from the data')

            drop_list = X_data[X_data['thal'].isna()].index.to_list()
            f_log_obj.warning('The indices {} of X_data and y_data are dropped\n'.format(drop_list))
            X_data.drop(labels=drop_list, axis=0, inplace=True)
            y_data.drop(labels=drop_list, axis=0, inplace=True)

        else:
            f_log_obj.debug('There is an unexpected feature "{}"'.format(i))
            f_log_obj.critical('This unexpected feature most likely breaks the code further down')


    # Reset the indices of X and y data
    X_data.reset_index(inplace=True, drop=True)
    y_data.reset_index(inplace=True, drop=True)

    # Change target values to present or absence of heart disease
    y_data.loc[y_data['num'] > 0] = 1
    y_data.rename(columns={'num': 'target'}, inplace=True)

    # Set the SettingWithCopyWarnings to "warn" again
    pd.options.mode.chained_assignment = 'warn'

    # Do the return statement here for X_data and y_data
    return X_data, y_data


def visualize(visual_obj, f_path, p_log_obj):
    """
    Visualize Data as EDA to look for outliers
    :param visual_obj: obj, Instance of Visualizer class from X_data and y_data set
    :param f_path: str, Path to figures directory
    :param p_log_obj: obj, Instance of Logger class for preprocessing
    :return: no return value
    """

    visual_obj.pairplot()
    visual_obj.save_pic(f_path)
    visual_obj.correlation()
    visual_obj.save_pic(f_path)
    visual_obj.barplot('sex')
    visual_obj.save_pic(f_path)
    visual_obj.barplot('fbs')
    visual_obj.save_pic(f_path)
    visual_obj.barplot('exang')
    visual_obj.save_pic(f_path)
    visual_obj.barplot('cp')
    visual_obj.save_pic(f_path)
    visual_obj.boxplot()
    visual_obj.save_pic(f_path)

    p_log_obj.info('The features will not be further pre-processed.')
    p_log_obj.info('Dataset will be stored in directory "data".')
    p_log_obj.warning('Outliers in some features may affect performance of model. This has to be checked later on.')