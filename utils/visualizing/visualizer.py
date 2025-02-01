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

        # Create column with correct target names
        self.data['target_name'] = self.data['target'].map(str)
        self.data['target_name'] = self.data['target_name'].str.replace('0', 'Normal')
        self.data['target_name'] = self.data['target_name'].str.replace('1', 'Diseased')
        self.data['target_name'] = self.data['target_name'].astype('category')

        self.hue = 'target_name'
        self.pic_name = None


    def __str__(self):
        """Print statement returns properties of data like the shape and the feature names"""
        return f'\nThe current dataset has the following properties:\n' \
               f'- shape: {self.data.shape}\n' \
               f'- feature-names: {self.data.columns.to_list()}'


    def pairplot(self):
        """Returns a sns pairplot of all numeric features of the data."""
        self.pic_name = 'Feature_pairplot.png'
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,9))
        sns.pairplot(self.data, hue=self.hue, hue_order=[0,1], markers=['X', 'o'])
        ax.set_title('Pairplot of numeric features')
        ax.title.set_size(20)


    def correlation(self):
        """Creates linear correlation plot in a heatmap"""
        self.pic_name = 'Feature_correlation.png'
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,9))
        correlation = self.data.corr()
        sns.heatmap(correlation, annot=True, fmt='.2f', linewidth=.5)
        ax.set_title('Correlation map of features')
        ax.title.set_size(20)


    def barplot(self, f_name):
        """
        Creates a barplot of the different values of the feature
        :param f_name: str, feature name which will be visualized
        """
        lookup = {'sex': {'labels': ['Female', 'Male'], 'title': 'Diseased Female vs Male'},
                  'fbs': {'labels': ['False', 'True'], 'title': 'Fasting Blood sugar'},
                  'exang': {'labels': ['No', 'Yes'], 'title': 'Exercise induced Angina'},
                  'cp': {'labels': ['Typical angina', 'Atypical angina', 'Non-Aningal pain', 'Asymptomatic'], 'title': 'Chest Pain Types'}}
        self.pic_name = f_name + '_countplot.png'

        g = sns.catplot(self.data, x=f_name, hue=self.hue, kind='count', errorbar='ci', height=5, aspect=1.5, legend=False)
        g.set_xticklabels(lookup[f_name]['labels'])
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(lookup[f_name]['title'])
        plt.legend(title='Target', loc='upper left', labels=self.data['target_name'])


    def boxplot(self):
        """Create boxplots of categorical features"""
        



    def save_pic(self, path_to_file):
        """In order to save the current picture this function is called"""
        if self.pic_name:
            plt.savefig(os.path.join(path_to_file,self.pic_name))
            self.pic_name = None