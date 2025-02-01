import os

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
        self.pic_name = None


    def __str__(self):
        """Print statement returns properties of data like the shape and the feature names"""
        return f'\nThe current dataset has the following properties:\n' \
               f'- shape: {self.data.shape}\n' \
               f'- feature-names: {self.data.columns.to_list()}'


    def pairplot(self):
        """Returns a sns pairplot of all numeric features of the data."""
        self.pic_name = 'Feature_pairplot.png'
        plt.clf()
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,9))
        sns.pairplot(self.data, hue='target', hue_order=[0,1], markers=['X', 'o'])
        ax.set_title('Pairplot of numeric features')
        # Instead of show save the image
        #plt.show()


    def correlation(self):
        """Creates linear correlation plot in a heatmap"""
        self.pic_name = 'Feature_correlation.png'
        plt.clf()
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,9))
        correlation = self.data.corr()
        sns.heatmap(correlation, annot=True, fmt='.2f', linewidth=.5)
        ax.set_title('Correlation map of features')
        ax.title.set_size(20)
        # Instead of show save the image
        #plt.show()


    def boxplot(self):
        pass


    def save_pic(self, path_to_file):
        """In order to save the current picture this function is called"""
        if self.pic_name:
            plt.savefig(os.path.join(path_to_file,self.pic_name))
            self.pic_name = None