from typing import Optional, List

import ConfigSpace
import numpy as np

import sklearn.impute
from ConfigSpace import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter, Constant
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class SurrogateModel:

    def __init__(self, config_space):
        self.config_space = config_space
        self.df = None
        self.model = None

        self.trained_cols: List[str] = []
        self.max_anchor_size_: Optional[float] = None

        try:
            hps = self.config_space.get_hyperparameters()
        except AttributeError:
            try:
                hps = list(self.config_space.values())
            except Exception:
                hps = [self.config_space.get_hyperparameter(k) for k in list(self.config_space.keys())]

        self._all_hp_names = [hp.name for hp in hps]

        self._cat_hp_names = [hp.name for hp in hps if self.is_categorical(hp)]
        self._num_hp_names = [hp.name for hp in hps if self.is_numeric(hp)]

    @staticmethod
    def is_categorical(hp) -> bool:
        return isinstance(hp, CategoricalHyperparameter)

    @staticmethod
    def is_numeric(hp) -> bool:
        return isinstance(hp, (UniformIntegerHyperparameter, UniformFloatHyperparameter, Constant))


    def fit(self, df):
        """
        Receives a data frame, in which each column (except for the last two) represents a hyperparameter, the
        penultimate column represents the anchor size, and the final column represents the performance.

        :param df: the dataframe with performances
        :return: Does not return anything, but stores the trained model in self.model
        """
        self.df = df.copy()
        y = self.df["score"].to_numpy()
        hp_cols_present = [c for c in self._all_hp_names if c in self.df.columns]
        feature_cols = hp_cols_present + ["anchor_size"]
        X = self.df[feature_cols].copy()
        self.max_anchor_size_ = float(self.df["anchor_size"].max())

        category_columns = [c for c in self._cat_hp_names if c in X.columns]
        numerical_columns = [c for c in self._num_hp_names if c in X.columns]

        if "anchor_size" not in numerical_columns:
            numerical_columns = numerical_columns + ["anchor_size"]

        # final_anchor = df['anchor_size'].max()
        # df_final = df.loc[df["anchor_size"] == final_anchor].copy()
        # y = df_final["score"].to_numpy()
        # X = df_final.drop(columns=["score", "anchor_size"])
        #
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #
        # category_columns = ["metric", "pp@cat_encoder", "pp@decomposition", "pp@featuregen", "pp@featureselector",
        #                     "pp@scaler", "weights", "pp@kernel_pca_kernel", "pp@std_with_std"]
        # numerical_cols = [c for c in X.columns if c not in category_columns]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore"))
                ]), category_columns),
                ("num", SimpleImputer(strategy="median"), numerical_columns),
            ],
            remainder="drop",
        )

        rf = RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_split=4, min_samples_leaf=2,
                                   max_features=0.5, bootstrap=True, n_jobs=-1, random_state=0)

        self.model = Pipeline([
            ("preprocessor", preprocessor),
            ("model", rf)
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.trained_cols = list(X.columns)


        print(self.model.score(X_test, y_test))
        print(mean_squared_error(y_pred, y_test))
        print(r2_score(y_pred, y_test))

    def predict(self, theta_new):
        """
        Predicts the performance of a given configuration theta_new

        :param theta_new: a dict, where each key represents the hyperparameter (or anchor)
        :return: float, the predicted performance of theta new (which can be considered the ground truth)
        """

        X_new = pd.DataFrame([theta_new]).copy()
        if "anchor_size" not in X_new.columns:
            if self.max_anchor_size_ is None:
                raise RuntimeError("Model has no recorded max anchor size; fit the model first.")
            X_new["anchor_size"] = self.max_anchor_size_

        for trained_col in self.trained_cols:
            if trained_col not in X_new.columns:
                X_new[trained_col] = np.nan

        X_new = X_new[self.trained_cols]
        prediction = self.model.predict(X_new)[0]
        return float(prediction)
