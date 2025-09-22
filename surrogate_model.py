import ConfigSpace

import sklearn.impute
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import pandas as pd


class SurrogateModel:

    def __init__(self, config_space):
        self.config_space = config_space
        self.df = None
        self.model = None

    def fit(self, df):
        """
        Receives a data frame, in which each column (except for the last two) represents a hyperparameter, the
        penultimate column represents the anchor size, and the final column represents the performance.

        :param df: the dataframe with performances
        :return: Does not return anything, but stores the trained model in self.model
        """

        self.df = df
        final_anchor = self.df['anchor_size'].max()
        print(final_anchor)
        self.df = df[self.df['anchor_size'] == final_anchor]
        self.df.drop('anchor_size', axis=1, inplace=True)
        X = self.df.drop('score', axis=1)
        y = self.df['score']
        print(X.shape)
        print(y)
        self.model = RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_split=4, min_samples_leaf=2,
                                           max_features=0.5, bootstrap=True, oob_score=True, n_jobs=-1,
                                           random_state=0) if self.model is None else self.model
        self.model.fit(X, y)

    def predict(self, theta_new):
        """
        Predicts the performance of a given configuration theta_new

        :param theta_new: a dict, where each key represents the hyperparameter (or anchor)
        :return: float, the predicted performance of theta new (which can be considered the ground truth)
        """
        raise NotImplementedError()
