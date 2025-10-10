import numbers

import ConfigSpace
import numpy as np
from typing import List, Tuple, Dict

import pandas as pd
from ConfigSpace import Configuration, UniformIntegerHyperparameter, UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant, OrdinalHyperparameter
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class SequentialModelBasedOptimization(object):

    def __init__(self, config_space: ConfigSpace):
        """
        Initializes empty variables for the model, the list of runs (capital R), and the incumbent
        (theta_inc being the best found hyperparameters, theta_inc_performance being the performance
        associated with it)
        """
        RANDOM_STATE = 42

        self.config_space = config_space
        self.R: list[tuple[ConfigSpace.Configuration, float]] = []

        self.theta_inc: ConfigSpace.Configuration | None = None
        self.theta_inc_performance: float | None = None

        self.pre = self.get_model_preprocessor(self.config_space)

        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-6)
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=RANDOM_STATE,
            alpha=0.0
        )
        self.model_pipeline = Pipeline([
            ("pre", self.pre),
            ("gpr", gpr),
        ])
        self.numeric_cols, self.categorical_cols = self.split_hyperparameters(self.config_space)
        self.all_cols = self.numeric_cols + self.categorical_cols
        self.rng = np.random.default_rng(RANDOM_STATE)

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
        # try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        # except TypeError:
        #     onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

        numeric_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
        ])
        categorical_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", onehot),
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

    # @staticmethod
    # def config_to_array(config: ConfigSpace.Configuration) -> np.ndarray:
    #     arr = config.get_array()
    #     return np.asarray(arr, dtype=float).reshape(1, -1)

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
        self.R = [(Configuration(self.config_space, values=self.filter_configs(config)), float(performance)) for
                  config, performance in
                  capital_phi]
        best_idx = int(np.argmin([performance for _, performance in self.R]))
        self.theta_inc, self.theta_inc_performance = self.R[best_idx]
        self.fit_model()

    def get_row(self, config: dict | ConfigSpace.Configuration) -> dict:
        config_dict = config.get_dictionary() if isinstance(config, Configuration) else dict(config)
        return {col: config_dict.get(col, None) for col in self.all_cols}

    # TO FIX
    def fit_model(self) -> None:
        """
        Fits the internal surrogate model on the complete run list.
        """
        # X = np.vstack([self.config_to_array(config) for config, _ in self.R])
        rows = [self.get_row(cfg) for cfg, _ in self.R]
        X = pd.DataFrame(rows, columns=self.all_cols)
        y = np.asarray([performance for _, performance in self.R], dtype=float)
        self.model_pipeline.fit(X, y)

    # TO FIX
    def select_configuration(self, n_configurations: int = 512) -> ConfigSpace.Configuration:
        """
        Determines which configurations are good, based on the internal surrogate model.
        Note that we are minimizing the error, but the expected improvement takes into account that.
        Therefore, we are maximizing expected improvement here.

        :return: A size n vector, same size as each element representing the EI of a given
        configuration
        """
        # candidate_configs = [self.config_space.sample_configuration() for _ in range(n_configurations)]
        # X_candidate = np.vstack([self.config_to_array(config) for config in candidate_configs])

        candidate_configs = [self.config_space.sample_configuration() for _ in range(n_configurations)]
        candidate_rows = [self.get_row(config) for config in candidate_configs]
        X_candidate = pd.DataFrame(candidate_rows, columns=self.all_cols)

        expected_improvement = self.expected_improvement(self.model_pipeline, self.theta_inc_performance, X_candidate)

        # Can remove order bias for ties

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

    # TO FIX
    def update_runs(self, run: Tuple[Dict, float]):
        """
        After a configuration has been selected and ran, it will be added to the run list
        (so that the model can be trained on it during the next iterations).

        :param run: A tuple (configuration, performance) where performance is error rate
        """
        config = Configuration(self.config_space, values=self.filter_configs(run[0]))
        performance = float(run[1])
        self.R.append((config, performance))

        if performance < self.theta_inc_performance:
            self.theta_inc = config
            self.theta_inc_performance = performance

        self.fit_model()
