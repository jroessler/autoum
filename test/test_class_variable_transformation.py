import unittest

import numpy as np
from sklearn.model_selection import train_test_split

from autouplift.approaches.class_variable_transformation import ClassVariableTransformation
from autouplift.approaches.utils import ApproachParameters, DataSetsHelper
from autouplift.datasets.utils import get_data_home, get_hillstrom_women_visit


class TestClassVariableTransformation(unittest.TestCase):

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

        self.cvt_parameters = {
            'n_estimators': 5,
            'max_depth': 5,
            'max_features': 5,
            'random_state': 123,
            "n_jobs": 10
        }

    def test_analyze(self):
        cvt = ClassVariableTransformation(self.cvt_parameters, self.approach_params)
        dict_scores = cvt.analyze(self.ds_helper)

        self.check_scores(dict_scores["score_train"], self.df_train)
        self.check_scores(dict_scores["score_valid"], self.df_valid)
        self.check_scores(dict_scores["score_test"], self.df_test)

    def test_transform(self):
        cvt = ClassVariableTransformation(self.cvt_parameters, self.approach_params)

        y = np.array([0, 0, 1, 1])
        t = np.array([0, 1, 0, 1])
        y_trans = np.array([1, 0, 0, 1])

        self.assertEqual(cvt.transform(y, t).tolist(), y_trans.tolist())

    def check_scores(self, scores, df):
        # Check if the number of scores is equal to the number of rows
        self.assertEqual(len(scores), df.shape[0])
        # Check if scores is an nd.array
        self.assertIsInstance(scores, np.ndarray)
        # Check if the scores are all >= -1
        self.assertGreaterEqual(scores.min(), -1)
        # Check if the scores are all <= 1
        self.assertLessEqual(scores.max(), 1)
