import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from autoum.approaches.lais_generalization import LaisGeneralization
from autoum.approaches.utils import ApproachParameters, DataSetsHelper
from autoum.datasets.utils import get_data_home, get_hillstrom_women_visit


class TestLaisGeneralization(unittest.TestCase):

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
        approach_params = ApproachParameters(cost_sensitive=False, feature_importance=False, path=root, post_prune=False, save=False, split_number=0)
        self.ds_helper = ds_helper
        self.approach_params = approach_params

        self.lais_parameters = {
            'n_estimators': 5,
            'max_depth': 5,
            'max_features': 5,
            'random_state': 123,
            "n_jobs": 10
        }

    def test_analyze_lais(self):
        lais = LaisGeneralization(self.lais_parameters, self.approach_params)
        dict_scores = lais.analyze(self.ds_helper)

        self.check_scores(dict_scores["score_train"], self.df_train)
        self.check_scores(dict_scores["score_valid"], self.df_valid)
        self.check_scores(dict_scores["score_test"], self.df_test)

    def test_transform_lais(self):
        lais = LaisGeneralization(self.lais_parameters, self.approach_params)

        df = pd.DataFrame(data={
            'treatment': [0, 1, 0, 1],
            'response': [0, 0, 1, 1]
        })
        correct_transforms = np.array([1, 2, 3, 4])
        self.assertEqual(lais.transform(df).tolist(), correct_transforms.tolist())

    def check_scores(self, scores, df):
        # Check if the number of scores is equal to the number of rows
        self.assertEqual(len(scores), df.shape[0])
        # Check if scores is an nd.array
        self.assertIsInstance(scores, np.ndarray)
