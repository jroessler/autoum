import logging
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 10)

from autoum.pipelines.pipeline_rw import PipelineRW
from autoum.const.const import *


class PipelineSD(PipelineRW):
    """
    Pipeline with synthetic data sets

    The synthetic data is created based on the method suggest by Wager & Athey, 2018, Estimation and Inference of Heterogeneous Treatment Effects using Random Forests, p.1238

    Either use k-fold Cross Validation or regular training/test split
    """

    def __init__(self, n: int, p: int, sigma: float, threshold: float, propensity: float, **kwargs):
        # Inheritance from parent class (PipelineRW)
        super().__init__(**kwargs)

        self.n = n
        self.p = p
        self.sigma = sigma
        self.threshold = threshold
        self.propensity = propensity

    def analyze_dataset(self, data: pd.DataFrame = None):
        """
        Apply, compare, and evaluate various uplift modeling approaches on synthetic data

        :param data: Dataset to be analyzed. Default: None as we want to analyze a synthetic dataset
        """
        logging.info(f"run_name: {self.run_name}, run_id: {self.run_name}, sigma: {self.sigma}, p: {self.p}, threshold: {self.threshold}, propensity: {self.propensity}")

        # Set random seed
        np.random.seed(self.random_seed)

        # Set start time
        start = time.time()

        # Create synthetic dataset
        df = self.create_synthetic_dataset()

        df_train, df_test = train_test_split(df, test_size=self.test_size, shuffle=True, stratify=df[['response', 'treatment']], random_state=self.random_seed)
        df_train.reset_index(inplace=True, drop=True)
        df_test.reset_index(inplace=True, drop=True)

        feature_names = list(df.drop(['response', 'treatment'], axis=1).columns.values)

        # If self.max_features is set to auto, set the number of features for the direct uplift approach to sqrt(n_features)
        if self.max_features == 'auto':
            n_features = len(feature_names)
            self.parameters[URF_TITLE + '_parameters']['max_features'] = int(round(np.sqrt(n_features)))

        # k-fold cross validation
        if self.cv_number_splits > 2:
            dict_uplift = self.analyze_k_fold_cv(df_train=df_train, df_test=df_test)

        # regular train/test split
        else:
            dict_uplift = self.analyze_single_fold(df_train=df_train, df_test=df_test)

        # Calculate, plot and save metrics
        self.calculate_metrics(list_feature_importances=dict_uplift["feature_importances"], feature_names=feature_names,
                               list_dict_uplift_train=dict_uplift["list_dict_uplift_train"], list_dict_uplift_valid=dict_uplift["list_dict_uplift_valid"],
                               list_dict_uplift_test=dict_uplift["list_dict_uplift_test"], list_dict_opt_uplift_train=dict_uplift["list_dict_opt_uplift_train"],
                               list_dict_opt_uplift_valid=dict_uplift["list_dict_opt_uplift_valid"], list_dict_opt_uplift_test=dict_uplift["list_dict_opt_uplift_test"])

        end = time.time()
        logging.info('Function took %0.3f s' % (end - start))

    def create_synthetic_dataset(self):
        """
        Create a complex, non-linear CATE function

        :return: DataFrame (synthetic data)
        """
        means = np.zeros(self.p)
        cov_mat = np.ones([self.p, self.p]) * 0.3 - (np.eye(self.p) * (0.3 - 1))
        x = np.random.multivariate_normal(means, cov_mat, size=self.n)
        e = np.repeat(self.propensity, self.n)
        w = np.random.binomial(1, e, size=self.n).reshape((self.n, -1))
        x = np.append(x, w, axis=1)

        def rho(x):
            return 2 / (1 + np.exp(-12 * (x - (1 / 2))))

        mu_1 = 1 / 2 * rho(x[:, 0]) * rho(x[:, 1])
        mu_0 = -1 / 2 * rho(x[:, 0]) * rho(x[:, 1])
        y_1 = mu_1 + self.sigma * np.random.normal(size=self.n)
        y_0 = mu_0 + self.sigma * np.random.normal(size=self.n)
        y = np.where(x[:, -1] == 1, y_1, y_0)
        y_binary = np.where(y > np.percentile(y, self.threshold), 1, 0)
        x = x[:, :-1]

        columns = [f"Feature{i}" for i in range(0, x.shape[1])]
        df = pd.DataFrame(data=x, columns=columns)
        df["response"] = y_binary
        df["treatment"] = w

        return df
