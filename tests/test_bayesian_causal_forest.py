import unittest

import numpy as np
from sklearn.model_selection import train_test_split

from autouplift.approaches.bayesian_causal_forest import BayesianCausalForest
from autouplift.approaches.utils import ApproachParameters, DataSetsHelper
from autouplift.datasets.utils import get_data_home, get_hillstrom_women_visit


class TestBayesianCausalForest(unittest.TestCase):

    def setUp(self):
        # Get data
        data = get_hillstrom_women_visit()
        data = data.sample(frac=0.5, random_state=123)
        df_train, df_test = train_test_split(data, test_size=0.2, shuffle=True, random_state=123)
        df_train, df_valid = train_test_split(df_train, test_size=0.2, shuffle=True, random_state=123)

        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test

        ds_helper = DataSetsHelper(df_train=df_train, df_valid=df_valid, df_test=df_test)
        root = f"{get_data_home()}/testing/models/"
        approach_params = ApproachParameters(cost_sensitive=False, feature_importance=False, path=root, save=False, split_number=0)
        self.ds_helper = ds_helper
        self.approach_params = approach_params

        self.bcf_parameters = {
            "num_sweeps": 50,
            "burnin": 15,
            "num_cutpoints": 100,
            "Nmin": 50,
            "max_depth": 5,
            "parallel": True,
            "standardize_target": False,
            "set_random_seed": True,
            "num_trees_pr": 5,
            "alpha_pr": 0.95,
            "beta_pr": 2,
            "num_trees_trt": 5,
            "alpha_trt": 0.95,
            "beta_trt": 2,
        }

    def test_bcf_analyze(self):
        bcf = BayesianCausalForest(self.bcf_parameters, self.approach_params)
        dict_scores = bcf.analyze(self.ds_helper)

        self.check_scores(dict_scores["score_train"], self.df_train)
        self.check_scores(dict_scores["score_valid"], self.df_valid)
        self.check_scores(dict_scores["score_test"], self.df_test)

    def check_scores(self, scores, df):
        # Check if the number of scores is equal to the number of rows
        self.assertEqual(len(scores), df.shape[0])
        # Check if scores is an nd.array
        self.assertIsInstance(scores, np.ndarray)
        # Check if the scores are all >= -1
        self.assertGreaterEqual(scores.min(), -1)
        # Check if the scores are all <= 1
        self.assertLessEqual(scores.max(), 1)
