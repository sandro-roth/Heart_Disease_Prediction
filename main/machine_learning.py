# Pre-training
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold

# Model selections
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# Hyper-parameter tuning



# model evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



class MachineLearning:
    """ML Class for classification algorithms to predict the presence of a heart disease"""
    def __init__(self, X_data, y_data, yml_obj):
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
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        return X_train_scaled, X_test_scaled


    def log_reg(self, path):
        """
        Performing Logistic Regression on the Classification problem
        :param path: str, directory of results
        :return: (str, obj), accuracy score of model and classification report of model
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
        plt.clf()

        # Evaluate accuracy on test set
        log_model.fit(X_train, self.y_train)
        y_pred = log_model.predict(X_test)
        acc_score = accuracy_score(self.y_test, y_pred)
        class_rep = classification_report(self.y_test, y_pred)
        g = sns.heatmap(confusion_matrix(self.y_test, y_pred), fmt='.2g', linewidths=0.5, annot=True)
        g.set(xlabel='Actual pathology', ylabel='Predicted pathology', xticklabels=['Normal', 'Diseased'],
              yticklabels=['Normal', 'Diseased'], title='Confusion matrix Logistic Regression')
        plt.savefig(path+'/confusion_matrix.png')
        return acc_score, class_rep


    def k_nearest(self):
        # use standardize method for data
        X_train, X_test = self.standardize()
        pass


    def random_forest(self):
        pass


    def g_boosted(self):
        pass