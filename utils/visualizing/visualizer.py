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
        self.hue = 'target'
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
        plt.legend(title='Target', loc='upper left', labels=self.data['target'])


    def boxplot(self):
        """Create boxplots of categorical features"""
        self.pic_name = 'Feature_boxplots.png'
        melted_df = pd.melt(self.data[['age', 'trestbps', 'chol', 'thalach']])

        # Separate the data for 'chol' and other features
        chol_df = melted_df[melted_df['variable'] == 'chol']
        other_df = melted_df[melted_df['variable'] != 'chol']

        # Create subplots with adjusted widths
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})

        # Plot for non-'chol' features
        sns.boxplot(
            data=other_df,
            x='variable', y='value',
            ax=axes[0],
            hue='variable',
            palette="Set2",
            linewidth=1.5,
            legend=False,
            width=0.4  # Adjust box width for a sleeker look
        )
        axes[0].set_title('Distribution of Age, Trestbps, and Thalach', fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Feature", fontsize=10, labelpad=15)
        axes[0].set_ylabel("Value", fontsize=10)
        axes[0].tick_params(axis='x', rotation=15)

        # Plot for 'chol'
        sns.boxplot(
            data=chol_df,
            x='variable', y='value',
            ax=axes[1],
            color='steelblue',
            linewidth=1.5,
            width=0.4  # Adjust box width for a sleeker look
        )
        axes[1].set_title('Distribution of Chol', fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Feature", fontsize=10, labelpad=15)
        axes[1].set_ylabel("Value", fontsize=10)
        axes[1].tick_params(axis='x', rotation=15)

        # Adjust layout for better spacing
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle('Data distribution of numerical features', fontsize=14, fontweight='bold')


    def save_pic(self, path_to_file):
        """In order to save the current picture this function is called"""
        if self.pic_name:
            plt.savefig(os.path.join(path_to_file,self.pic_name))
            self.pic_name = None