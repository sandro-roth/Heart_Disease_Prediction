class Visualizer:
    '''Visualizer feeded preprocessed data using multiple functions to display.'''
    def __init__(self, f_df, t_df):
        """
        :param f_df: feature pandas data frame, holding all the features to be analyzed
        :param t_df: target pandas data frame, holding only the target values
        """
        self.f_df = f_df
        self.t_df = t_df

    def descirbe(self):
        pass

    def __str__(self):
        pass

    def boxplot(self):
        pass
