import numpy as np
from ConfigSpace import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter, Constant
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def _normalize_categorical_to_object_and_nan(series: pd.Series) -> pd.Series:
    """Return series as object dtype with strings; missing → np.nan."""
    def _norm(v):
        if pd.isna(v):
            return np.nan
        return str(v)
    return series.astype("object").map(_norm)


class SurrogateModel:

    def __init__(self, config_space, seed = 0) -> None:
        self.config_space = config_space
        self.seed = seed
        self.df = None
        self.model = None

        self.trained_cols = []
        self.max_anchor_size = None

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

        for c in category_columns:
            if c in X.columns:
                X[c] = _normalize_categorical_to_object_and_nan(X[c])

        # OneHot dense across sklearn versions
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        preprocessor = ColumnTransformer(
            transformers=[
                # default missing_values=np.nan works because we enforce NaN above
                ("cat", Pipeline([
                    ("impute", SimpleImputer(strategy="constant", fill_value="__NA__")),
                    ("ohe", ohe),
                ]), category_columns),
                ("num", SimpleImputer(strategy="median"), numerical_columns),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )


        rf = RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_split=4, min_samples_leaf=2,
                                   max_features=0.5, bootstrap=True, n_jobs=-1, random_state=self.seed)

        self.model = Pipeline([
            ("preprocessor", preprocessor),
            ("model", rf)
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed)

        self.model.fit(X_train, y_train)
        self.trained_cols = list(X.columns)

    def predict(self, theta_new):
        """
        Predicts the performance of a given configuration theta_new

        :param theta_new: a dict, where each key represents the hyperparameter (or anchor)
        :return: float, the predicted performance of theta new (which can be considered the ground truth)
        """
        X_new = pd.DataFrame([theta_new]).copy()

        # normalize categoricals same as in fit()
        # Make sure categoricals match training-time convention
        for cname in getattr(self, "_cat_hp_names", []):
            if cname in X_new.columns:
                X_new[cname] = _normalize_categorical_to_object_and_nan(X_new[cname])



        if "anchor_size" not in X_new.columns:
            if getattr(self, "max_anchor_size_", None) is None:
                raise RuntimeError("SurrogateModel not fitted or max anchor unknown.")
            X_new["anchor_size"] = self.max_anchor_size_

        for trained_col in self.trained_cols:
            if trained_col not in X_new.columns:
                X_new[trained_col] = np.nan

        X_new = X_new[self.trained_cols]

        prediction = self.model.predict(X_new)[0]
        return float(prediction)
