import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Visualizer:
    """Visualizer feeded preprocessed data using multiple functions to display."""
    def __init__(self, f_df, t_df):
        """
        :param f_df: feature pandas data frame, holding all the features to be analyzed
        :param t_df: target pandas data frame, holding only the target values
        """
        self.f_df = f_df
        self.t_df = t_df
        self.data = pd.concat([f_df, t_df], axis=1)


    def __str__(self):
        """Print statement returns properties of data like the shape and the feature names"""
        return f'The current dataset has the following properties:\n' \
               f'shape: {self.data.shape}\n' \
               f'feature-names: {self.data.columns.to_list()}'

    def pairplot(self):
        """Returns a sns pairplot of all numeric features of the data."""
        plt.clf()
        sns.pairplot(self.data, hue='num', hue_order=[0,1], markers=['X', 'o'])
        plt.show()


    def correlation(self):
        pass


    def boxplot(self):
        pass
