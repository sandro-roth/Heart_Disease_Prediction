import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree


class MachineLearning:
    """
    ML Class for classification algorithms to predict the presence of a heart disease
    """
    def __init__(self, X_data, y_data, yml_obj):
        """"""
        self.f_name = X_data.columns
        X_data = X_data.values
        y_data = y_data.values.ravel()
        self.rs = yml_obj['ML']['random_state']
        ts = yml_obj['ML']['t_size']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_data, y_data,
                                                                                test_size=ts,
                                                                                stratify=y_data,
                                                                                random_state=self.rs)
        self.yml_obj = yml_obj


    def standardize(self):
        """
        Centering and Scaling the training data and applies transformation on the test data set.
        :return: (np.ndarray, np.ndarray), standardized data for training and testing
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        return X_train_scaled, X_test_scaled


    def heatmap(self, y_prediction, dir_path, fig_title):
        """
        Saves a heatmap from the prediction of the ML algorithm
        :param y_prediction: numpy-array, holds a models prediction of the target
        :param dir_path: str, path to store the figure to
        :param fig_title: str, title of the figure to be stored
        """
        plt.clf()
        g = sns.heatmap(confusion_matrix(self.y_test, y_prediction), fmt='.2g', linewidths=0.5, annot=True)
        g.set(xlabel='Actual pathology', ylabel='Predicted pathology', xticklabels=['Normal', 'Diseased'],
              yticklabels=['Normal', 'Diseased'], title=fig_title)
        plt.savefig(dir_path + '/confusion_matrix.png')


    def log_reg(self, path):
        """
        Performing Logistic Regression on the Classification problem
        :param path: str, directory of results
        :return: (float, str), accuracy score of model and classification report of model
        """
        log_model = LogisticRegression()

        # Parameter
        ns = self.yml_obj['log_reg']['KF_splits']
        ks = self.yml_obj['log_reg']['KF_shuffel']

        # Use standardize method for data
        X_train, X_test = self.standardize()

        # Use cross validation to evaluate training accuracy on different data sets
        kf = KFold(n_splits=ns, random_state=self.rs, shuffle=ks)
        cv_results = cross_val_score(log_model, X_train, self.y_train, cv=kf)
        plt.boxplot(cv_results, tick_labels=['Logistic Regression'])
        plt.title('6-Fold cross validation accuracy scores')
        plt.savefig(path+'/cvs_boxplot.png')

        # Evaluate accuracy on test set
        log_model.fit(X_train, self.y_train)
        y_pred = log_model.predict(X_test)
        acc_score = accuracy_score(self.y_test, y_pred)
        class_rep = classification_report(self.y_test, y_pred)
        self.heatmap(y_pred, path, 'Confusion matrix Logistic Regression')
        return acc_score, class_rep


    def k_nearest(self, path):
        """
        Performing k-Nearest Neighbors on the Classification problem
        :param path: str, directory of results
        :return: (float, str), accuracy score of model and classification report of model
        """
        knn = KNeighborsClassifier()

        # Parameter
        n = self.yml_obj['K_nearest']['Neighbors']
        m = self.yml_obj['K_nearest']['Metric']
        ns = self.yml_obj['K_nearest']['KF_splits']
        ks = self.yml_obj['K_nearest']['KF_shuffel']

        params_k = {'n_neighbors': n, 'metric': m}

        # Use standardize method for data
        X_train, X_test = self.standardize()

        # Hyperparameter tuning
        kf = KFold(n_splits=ns, shuffle=ks, random_state=self.rs)
        grid_k = GridSearchCV(estimator=knn, param_grid=params_k, cv=kf, scoring='accuracy', return_train_score=False)
        grid_k.fit(X_train, self.y_train)

        knn1 = grid_k.best_estimator_
        y_pred = knn1.predict(X_test)
        acc_score = accuracy_score(self.y_test, y_pred)
        class_rep = classification_report(self.y_test, y_pred)
        self.heatmap(y_pred, path, 'Confusion matrix k-Nearest neighbor')
        return grid_k, acc_score, class_rep


    def random_forest(self, path):
        """
        Performing Random Forest on the Classification problem
        :param path: str, directory of results
        :return: (float, str), accuracy score of model and classification report of model
        """
        rf_model = RandomForestClassifier(random_state=self.rs)

        # Parameter
        ns = self.yml_obj['r_forest']['KF_splits']
        ks = self.yml_obj['r_forest']['KF_shuffel']
        n_est = self.yml_obj['r_forest']['n_est']
        m_feat = self.yml_obj['r_forest']['max_features']
        m_dep = self.yml_obj['r_forest']['max_depth']
        cr = self.yml_obj['r_forest']['crit']

        # Hyperparameter tuning
        params_rf = {'n_estimators': n_est, 'max_features': m_feat, 'max_depth': m_dep, 'criterion': cr}
        kf = KFold(n_splits=ns, shuffle=ks, random_state=self.rs)
        grid_rf = GridSearchCV(estimator=rf_model, param_grid=params_rf, cv=kf, n_jobs=-1, scoring='accuracy')
        grid_rf.fit(self.X_train, self.y_train)

        rf_model1 = grid_rf.best_estimator_
        y_pred = rf_model1.predict(self.X_test)
        acc_score = accuracy_score(self.y_test, y_pred)
        class_rep = classification_report(self.y_test, y_pred)
        self.heatmap(y_pred, path, 'Confusion matrix Random Forest')

        # Check feature importance
        feature_imp = pd.Series(rf_model1.feature_importances_, index=self.f_name).sort_values()
        plt.clf()
        plt.figure(figsize=(8, 4))
        plt.barh(feature_imp.index, feature_imp.values, color='skyblue')
        plt.xlabel('Gini Importance')
        plt.title('Feature importance after hyper-parameter tuning')
        plt.gca().invert_yaxis()
        plt.savefig(path+'/feature_importance.png')

        # Save first instance of RandomForest
        fig, ax = plt.subplots(figsize=(4,4), dpi=800)
        tree.plot_tree(rf_model1.estimators_[99],
                       feature_names=self.f_name,
                       class_names=['Normal', 'Diseased'],
                       filled=True)
        fig.savefig(path+'/rf_individualtree.png')

        return grid_rf, acc_score, class_rep


    def g_boosted(self):
        pass


    def voting_class(self):
        pass