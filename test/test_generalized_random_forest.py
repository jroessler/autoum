import os
import sys
import unittest

import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()
root = os.getenv("ROOT_FOLDER")
sys.path.append(root + "src/")

from pipelines.helper.helper_pipeline import HelperPipeline
from approaches.helper.helper_approaches import ApproachParameters, DataSetsHelper

# This is the class we want to test in this file
from approaches.generalized_random_forest import GeneralizedRandomForest


class TestGeneralizedRandomForest(unittest.TestCase):

    def setUp(self):
        # Helper
        helper = HelperPipeline()

        # Dataset
        dataset_name = "Companye_k"

        df_train, df_test = helper.get_dataframe(dataset_name, 0.2, 123)
        df_train, df_valid = train_test_split(df_train, test_size=0.2, shuffle=True, random_state=123)

        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test

        ds_helper = DataSetsHelper(df_train=df_train, df_valid=df_valid, df_test=df_test)
        approach_params = ApproachParameters(cost_sensitive=False, feature_importance=False, path=root, save=False, split_number=0)
        self.ds_helper = ds_helper
        self.approach_params = approach_params

        self.grf_parameters = {
            "criterion": "het",
            "n_estimators": 20,
            "max_depth": 5,
            "min_samples_leaf": 50,
            "max_samples": 0.45,
            "discrete_treatment": False
        }

    def test_analyze(self):
        grf = GeneralizedRandomForest(self.grf_parameters, self.approach_params)
        dict_scores = grf.analyze(self.ds_helper)

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
