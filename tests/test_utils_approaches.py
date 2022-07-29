import unittest

import pandas as pd

# This is the class we want to test in this file
from autoum.approaches.utils import Helper


class TestHelperApproaches(unittest.TestCase):

    def test_create_class_weight(self):
        y = [0, 0, 0, 0, 1, 1]
        y_0_weight = 0.75
        y_1_weight = 1.5
        d_class_weights = Helper.create_class_weight(y)

        self.assertEqual(y_0_weight, d_class_weights[0])
        self.assertEqual(y_1_weight, d_class_weights[1])

    def test_add_treatment_group_key(self):
        data = [1, 1, 1, 0, 0, 0]
        df = pd.DataFrame(data={
            'treatment': data
        })

        experiment_groups_col_expected = ['t', 't', 't', 'c', 'c', 'c']
        experiment_groups_col = Helper.add_treatment_group_key(df)
        self.assertListEqual(experiment_groups_col_expected, experiment_groups_col.tolist())

    def test_get_propensity_score(self):
        length = 1000
        num_treatment_samples = 500
        propensity_score = Helper.get_propensity_score(length, num_treatment_samples)

        self.assertEqual(len(propensity_score), length)
        self.assertEqual(propensity_score.tolist().count((num_treatment_samples / length)), length)
