# Pre-training
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Model selections


# Hyper-parameter tuning


# model evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix



class MachineLearning:
    """ML Class for classification algorithms to predict the presence of a heart disease"""
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data



    def log_reg(self):
        pass

    def k_nearest(self):
        pass

    def random_forest(self):
        pass

    def g_boosted(self):
        pass