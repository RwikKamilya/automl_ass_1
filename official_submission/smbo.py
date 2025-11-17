import numbers

import warnings
from sklearn.exceptions import ConvergenceWarning

import ConfigSpace
import numpy as np
from typing import List, Tuple, Dict
from packaging import version

import pandas as pd
from ConfigSpace import Configuration, UniformIntegerHyperparameter, UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numbers
import sklearn

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(
    "ignore",
    message="Skipping features without any observed values",
    category=UserWarning
)

def _normalize_categorical_to_object_and_nan(series: pd.Series) -> pd.Series:
    """Return series as object dtype with strings; missing → np.nan."""
    def _norm(v):
        if pd.isna(v):
            return np.nan
        return str(v)
    return series.astype("object").map(_norm)


class SequentialModelBasedOptimization(object):

    def __init__(self, config_space: ConfigSpace, seed: int = 0):
        """
        Initializes empty variables for the model, the list of runs (capital R), and the incumbent
        (theta_inc being the best found hyperparameters, theta_inc_performance being the performance
        associated with it)
        """

        self.config_space = config_space
        self.seed = seed

        self.config_space.seed(self.seed)

        self.R = []
        self.theta_inc = None
        self.theta_inc_performance = None

        self.pre = self.get_model_preprocessor(self.config_space)

        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
                 + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))

        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,  # tiny jitter added to the diagonal
            normalize_y=True,  # centers the targets
            n_restarts_optimizer=3,  # restarts to escape poor local optima
            random_state=self.seed
        )

        self.model_pipeline = Pipeline([
            ("scaler", self.pre),
            ("gp", gpr)
        ])

        self.numeric_cols, self.categorical_cols = self.split_hyperparameters(self.config_space)
        self.all_cols = self.numeric_cols + self.categorical_cols
        self.rng = np.random.default_rng(self.seed)

    @staticmethod
    def split_hyperparameters(cs: ConfigSpace.ConfigurationSpace) -> Tuple[List[str], List[str]]:
        numeric_cols, categorical_cols = [], []

        for hp in cs.get_hyperparameters():
            name = hp.name

            if isinstance(hp, (UniformIntegerHyperparameter, UniformFloatHyperparameter)):
                numeric_cols.append(name)

            elif isinstance(hp, CategoricalHyperparameter):
                categorical_cols.append(name)

            elif isinstance(hp, Constant):
                if isinstance(hp.value, numbers.Number):
                    numeric_cols.append(name)
                else:
                    categorical_cols.append(name)
            else:
                categorical_cols.append(name)

        return numeric_cols, categorical_cols

    def get_model_preprocessor(self, cs: ConfigSpace.ConfigurationSpace) -> ColumnTransformer:
        numeric_cols, categorical_cols = self.split_hyperparameters(cs)

        # Dense output for GPR
        try:
            onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

        numeric_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ])
        categorical_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", onehot)
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_cols),
                ("cat", categorical_pipeline, categorical_cols),
            ],
            remainder="drop",
            sparse_threshold=0.0,  # force dense for GPR
            verbose_feature_names_out=False,
        )
        return preprocessor

    def filter_configs(self, values: dict) -> dict:
        valid_hyperparameters = set(self.config_space.get_hyperparameter_names())
        return {k: v for k, v in values.items() if k in valid_hyperparameters}

    def initialize(self, capital_phi: List[Tuple[Dict, float]]) -> None:
        """
        Initializes the model with a set of initial configurations, before it can make recommendations
        which configurations are in good regions. Note that we are minimising (lower values are preferred)

        :param capital_phi: a list of tuples, each tuple being a configuration and the performance (typically,
        error rate)
        """
        self.R = [
            (Configuration(self.config_space, values=self.filter_configs(config), allow_inactive_with_values=True),
             float(performance)) for config, performance in capital_phi]
        best_idx = int(np.argmin([performance for _, performance in self.R]))
        self.theta_inc, self.theta_inc_performance = self.R[best_idx]
        self.fit_model()

    def get_row(self, config: dict | ConfigSpace.Configuration) -> dict:
        if not isinstance(config, Configuration):
            config = Configuration(self.config_space,
                                   values=self.filter_configs(config),
                                   allow_inactive_with_values=True)
        config_dict = config.get_dictionary()
        return {col: config_dict.get(col, None) for col in self.all_cols}

    def fit_model(self) -> None:
        if not self.R:
            return

        rows = []
        for cfg, perf in self.R:
            rows.append((cfg.get_dictionary(), float(perf)))
        X = pd.DataFrame([r[0] for r in rows])
        y = np.array([r[1] for r in rows], dtype=float)

        # Split numeric vs categorical by coercibility
        numeric_cols, categorical_cols = [], []
        for c in X.columns:
            x_num = pd.to_numeric(X[c], errors="coerce")
            if x_num.notna().mean() >= 0.7:
                numeric_cols.append(c)
                X[c] = x_num.astype(float)         # numeric → float, NaN for missings
            else:
                categorical_cols.append(c)
                X[c] = _normalize_categorical_to_object_and_nan(X[c])

        ohe_kwargs = dict(handle_unknown="ignore")
        if version.parse(sklearn.__version__) >= version.parse("1.2"):
            ohe_kwargs["sparse_output"] = False
        else:
            ohe_kwargs["sparse"] = False

        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),             # expects NaN
            ("scaler", StandardScaler()),
        ])

        cat_pipe = Pipeline([
            # IMPORTANT: default missing_values=np.nan (what we pass)
            ("imputer", SimpleImputer(strategy="constant", fill_value="__NA__")),
            ("onehot", OneHotEncoder(**ohe_kwargs)),
        ])

        pre = ColumnTransformer(
            transformers=[
                ("num", num_pipe, numeric_cols),
                ("cat", cat_pipe, categorical_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
        model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=self.seed)

        self.model_pipeline = Pipeline([
            ("pre", pre),
            ("model", model),
        ])
        self.model_pipeline.fit(X, y)
        self.pre = pre
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols



    def select_configuration(self, n_configurations: int = 512) -> ConfigSpace.Configuration:
        """
        Determines which configurations are good, based on the internal surrogate model.
        Note that we are minimizing the error, but the expected improvement takes into account that.
        Therefore, we are maximizing expected improvement here.

        :return: A size n vector, same size as each element representing the EI of a given
        configuration
        """
        candidate_configs = [self.config_space.sample_configuration() for _ in range(n_configurations)]
        candidate_rows = [self.get_row(config) for config in candidate_configs]
        X_candidate = pd.DataFrame(candidate_rows, columns=self.all_cols)

        for c in X_candidate.columns:
            x_num = pd.to_numeric(X_candidate[c], errors="coerce")
            if c in getattr(self, "numeric_cols", []) or x_num.notna().mean() >= 0.7:
                X_candidate[c] = x_num.astype(float)  # numeric → float with NaN
            else:
                X_candidate[c] = _normalize_categorical_to_object_and_nan(X_candidate[c])


        expected_improvement = self.expected_improvement(self.model_pipeline, self.theta_inc_performance, X_candidate)
        pick = np.argmax(expected_improvement)
        return candidate_configs[pick]


    @staticmethod
    def expected_improvement(model_pipeline: Pipeline, f_star: float, theta: np.array) -> np.array:
        """
        Acquisition function that determines which configurations are good and which
        are not good.

        :param model_pipeline: The internal surrogate model (should be fitted already)
        :param f_star: The current incumbent (theta_inc)
        :param theta: A (n, m) array, each column represents a hyperparameter and each row
        represents a configuration
        :return: A size n vector, same size as each element representing the EI of a given
        configuration
        """
        low_sigma = 1e-12
        mu, sigma = model_pipeline.predict(theta, return_std=True)
        sigma = np.maximum(sigma, low_sigma)
        improvement = f_star - mu
        unit_improvement = np.clip(improvement / sigma, -8.0, 8.0)
        expected_improvement = improvement * norm.cdf(unit_improvement) + sigma * norm.pdf(unit_improvement)
        zero_sigma = sigma <= low_sigma
        if np.any(zero_sigma):
            expected_improvement[zero_sigma] = np.maximum(improvement[zero_sigma], 0.0)
        return expected_improvement

    def update_runs(self, run: Tuple[Dict, float]):
        """
        After a configuration has been selected and ran, it will be added to the run list
        (so that the model can be trained on it during the next iterations).

        :param run: A tuple (configuration, performance) where performance is error rate
        """
        config = Configuration(self.config_space, values=self.filter_configs(run[0]), allow_inactive_with_values=True)
        performance = float(run[1])
        self.R.append((config, performance))

        if self.theta_inc_performance is None or performance < self.theta_inc_performance:
            self.theta_inc = config
            self.theta_inc_performance = performance

        self.fit_model()
