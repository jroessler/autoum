import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from sklearn.model_selection import train_test_split

from autouplift.approaches.bayesian_causal_forest import BayesianCausalForest
from autouplift.approaches.class_variable_transformation import ClassVariableTransformation
from autouplift.approaches.generalized_random_forest import GeneralizedRandomForest
from autouplift.approaches.lais_generalization import LaisGeneralization
from autouplift.approaches.r_learner import RLearner
from autouplift.approaches.s_learner import SLearner
from autouplift.approaches.traditional import Traditional
from autouplift.approaches.treatment_dummy import TreatmentDummy
from autouplift.approaches.two_model import TwoModel
from autouplift.approaches.uplift_random_forest import UpliftRandomForest
from autouplift.approaches.utils import ApproachParameters, DataSetsHelper
from autouplift.approaches.x_learner import XLearner
from autouplift.const.const import *
from autouplift.datasets.utils import get_hillstrom_women_visit
from autouplift.pipelines.utils import HelperPipeline


class TestHelperPipeline(unittest.TestCase):

    def setUp(self):
        # Get data
        data = get_hillstrom_women_visit()
        self.data = data.sample(frac=0.5, random_state=123)
        self.df_train, self.df_test = train_test_split(data, test_size=0.2, shuffle=True, stratify=data[['response', 'treatment']], random_state=123)
        self.df_train, self.df_valid = train_test_split(data, test_size=0.2, shuffle=True, stratify=data[['response', 'treatment']], random_state=123)

        self.ds_helper = DataSetsHelper(df_train=self.df_train, df_valid=self.df_valid, df_test=self.df_test)
        self.approach_params = ApproachParameters(cost_sensitive=False, feature_importance=False, path=None, save=False, split_number=0)
        self.apply_params = {
            "data_set_helper": self.ds_helper,
            "feature_importance": False,
        }

        n_estimators = 100
        max_depth = 5
        max_features = 'auto'
        random_seed = 123
        min_samples_leaf = 5
        min_samples_treatment = 5
        n_jobs = 1

        urf_parameters = {
            "n_estimators": n_estimators,
            "max_features": None,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_treatment": min_samples_treatment,
            "n_reg": 100,
            "random_state": random_seed,
            "n_jobs": n_jobs,
            "control_name": "c",
            "normalization": True,
            "honesty": False
        }

        s_learner_parameters = {
            'n_estimators': n_estimators,
            "min_samples_leaf": min_samples_leaf,
            'max_depth': max_depth,
            'max_features': max_features,
            'random_state': random_seed,
            "n_jobs": n_jobs
        }

        traditional_parameters = {
            'n_estimators': n_estimators,
            "min_samples_leaf": min_samples_leaf,
            'max_depth': max_depth,
            'max_features': max_features,
            'random_state': random_seed,
            "n_jobs": n_jobs
        }

        cvt_parameters = {
            'n_estimators': n_estimators,
            "min_samples_leaf": min_samples_leaf,
            'max_depth': max_depth,
            'max_features': max_features,
            'random_state': random_seed,
            "n_jobs": n_jobs
        }

        lais_parameters = {
            'n_estimators': n_estimators,
            "min_samples_leaf": min_samples_leaf,
            'max_depth': max_depth,
            'max_features': max_features,
            'random_state': random_seed,
            "n_jobs": n_jobs
        }

        two_model_parameters = {
            'n_estimators': n_estimators,
            "min_samples_leaf": min_samples_leaf,
            'max_depth': max_depth,
            'max_features': max_features,
            'random_state': random_seed,
            "n_jobs": n_jobs
        }

        x_learner_parameters = {
            'n_estimators': n_estimators,
            "min_samples_leaf": min_samples_leaf,
            'max_depth': max_depth,
            'max_features': max_features,
            'random_state': random_seed,
            "n_jobs": n_jobs
        }

        r_learner_parameters = {
            'n_estimators': n_estimators,
            "min_samples_leaf": min_samples_leaf,
            'max_depth': max_depth,
            'max_features': max_features,
            'random_state': random_seed,
            "n_jobs": n_jobs
        }

        treatment_dummy_parameters = {
            'random_state': random_seed,
            "n_jobs": n_jobs,
            "max_iter": 10000
        }

        grf_parameters = {
            "criterion": "het",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_samples": 0.45,
            "discrete_treatment": False,
            "random_state": random_seed,
            "n_jobs": n_jobs
        }

        bcf_parameters = {  # BART parameters
            "num_sweeps": 50,
            "burnin": 15,
            "num_cutpoints": 100,
            "Nmin": min_samples_leaf,
            "max_depth": max_depth,
            "parallel": True,
            "standardize_target": False,
            "set_random_seed": True,
            "random_seed": random_seed,  # Prognostic BART parameters
            "num_trees_pr": n_estimators,
            "alpha_pr": 0.95,
            "beta_pr": 2,  # Treatment BART parameters
            "num_trees_trt": n_estimators,
            "alpha_trt": 0.95,
            "beta_trt": 2,
        }

        self.parameters = {
            URF_TITLE + "_parameters": urf_parameters,
            SLEARNER_TITLE + "_parameters": s_learner_parameters,
            TRADITIONAL_TITLE + '_parameters': traditional_parameters,
            CVT_TITLE + '_parameters': cvt_parameters,
            LAIS_TITLE + '_parameters': lais_parameters,
            TWO_MODEL_TITLE + '_parameters': two_model_parameters,
            XLEARNER_TITLE + '_parameters': x_learner_parameters,
            RLEARNER_TITLE + '_parameters': r_learner_parameters,
            TREATMENT_DUMMY_TITLE + '_parameters': treatment_dummy_parameters,
            GRF_TITLE + '_parameters': grf_parameters,
            BCF_TITLE + '_parameters': bcf_parameters
        }

    def test_apply_approaches(self):
        helper = HelperPipeline()

        result_dict = {
            "score_train": [],
            "score_valid": [],
            "score_test": [],
            "feature_importance": {}
        }

        # Create classifier
        classifier_list = [UpliftRandomForest(self.parameters[URF_TITLE + "_parameters"], self.approach_params, eval_function="ED"),
                           UpliftRandomForest(self.parameters[URF_TITLE + "_parameters"], self.approach_params, eval_function="KL"),
                           UpliftRandomForest(self.parameters[URF_TITLE + "_parameters"], self.approach_params, eval_function="Chi"),
                           UpliftRandomForest(self.parameters[URF_TITLE + "_parameters"], self.approach_params, eval_function="DDP"),
                           UpliftRandomForest(self.parameters[URF_TITLE + "_parameters"], self.approach_params, eval_function="IT"),
                           UpliftRandomForest(self.parameters[URF_TITLE + "_parameters"], self.approach_params, eval_function="CIT"),
                           UpliftRandomForest(self.parameters[URF_TITLE + "_parameters"], self.approach_params, eval_function="CTS"),
                           SLearner(self.parameters[SLEARNER_TITLE + "_parameters"], self.approach_params),
                           ClassVariableTransformation(self.parameters[CVT_TITLE + "_parameters"], self.approach_params),
                           LaisGeneralization(self.parameters[LAIS_TITLE + "_parameters"], self.approach_params),
                           TwoModel(self.parameters[TWO_MODEL_TITLE + "_parameters"], self.approach_params),
                           Traditional(self.parameters[TRADITIONAL_TITLE + "_parameters"], self.approach_params),
                           XLearner(self.parameters[XLEARNER_TITLE + "_parameters"], self.approach_params),
                           RLearner(self.parameters[RLEARNER_TITLE + "_parameters"], self.approach_params),
                           TreatmentDummy(self.parameters[TREATMENT_DUMMY_TITLE + "_parameters"], self.approach_params),
                           GeneralizedRandomForest(self.parameters[GRF_TITLE + "_parameters"], self.approach_params),
                           BayesianCausalForest(self.parameters[BCF_TITLE + "_parameters"], self.approach_params)]

        for classifier in classifier_list:
            with self.subTest(i=classifier):
                classifier.analyze = MagicMock(return_value=result_dict)
                *results, = helper.apply_approach(classifier, **self.apply_params)
                classifier.analyze.assert_called_once_with(self.ds_helper)

                self.assertIsInstance(results[0], list)
                self.assertIsInstance(results[1], list)
                self.assertIsInstance(results[2], list)
                self.assertIsInstance(results[3], dict)

                classifier.analyze.reset_mock()

    @patch('tests.test_utils_pipelines.HelperPipeline.apply_approach')
    def test_apply_uplift_approaches(self, m_apply_approach):
        helper = HelperPipeline()

        result_dict = ([], [], [], {})
        m_apply_approach.return_value = result_dict

        approaches = ['TWO_MODEL', 'URF_ED', 'URF_KL', 'URF_CHI', 'URF_DDP', 'URF_IT', 'URF_CIT', 'URF_CTS', 'TRADITIONAL', 'SLEARNER', 'CVT', 'LAIS', 'XLEARNER', 'RLEARNER',
                      'TREATMENT_DUMMY', 'GRF', 'BCF']
        for i in approaches:
            with self.subTest(i=i):
                result = helper.apply_uplift_approaches(self.df_train, self.df_valid, self.df_test, self.parameters, [i], split_number=0)

                self.assertEqual(result["df_scores_train"].shape[0], self.df_train.shape[0])
                self.assertEqual(result["df_scores_train"].shape[1], 3)
                self.assertEqual(result["df_scores_valid"].shape[0], self.df_valid.shape[0])
                self.assertEqual(result["df_scores_valid"].shape[1], 3)
                self.assertEqual(result["df_scores_test"].shape[0], self.df_test.shape[0])
                self.assertEqual(result["df_scores_test"].shape[1], 3)

                self.assertIsInstance(result["df_scores_train"], pd.DataFrame)
                self.assertIsInstance(result["df_scores_valid"], pd.DataFrame)
                self.assertIsInstance(result["df_scores_test"], pd.DataFrame)
                self.assertIsInstance(result["feature_importances"], dict)

                if i == "TWO_MODEL":
                    self.assertTrue(TwoModel.__instancecheck__(m_apply_approach.call_args[0][0]))
                elif "URF" in i:
                    self.assertTrue(UpliftRandomForest.__instancecheck__(m_apply_approach.call_args[0][0]))
                elif i == "TRADITIONAL":
                    self.assertTrue(Traditional.__instancecheck__(m_apply_approach.call_args[0][0]))
                elif i == "SLEARNER":
                    self.assertTrue(SLearner.__instancecheck__(m_apply_approach.call_args[0][0]))
                elif i == "CVT":
                    self.assertTrue(ClassVariableTransformation.__instancecheck__(m_apply_approach.call_args[0][0]))
                elif i == "LAIS":
                    self.assertTrue(LaisGeneralization.__instancecheck__(m_apply_approach.call_args[0][0]))
                elif i == "XLEARNER":
                    self.assertTrue(XLearner.__instancecheck__(m_apply_approach.call_args[0][0]))
                elif i == "RLEARNER":
                    self.assertTrue(RLearner.__instancecheck__(m_apply_approach.call_args[0][0]))
                elif i == "TREATMENT_DUMMY":
                    self.assertTrue(TreatmentDummy.__instancecheck__(m_apply_approach.call_args[0][0]))
                elif i == "GRF":
                    self.assertTrue(GeneralizedRandomForest.__instancecheck__(m_apply_approach.call_args[0][0]))
                elif i == "BCF":
                    self.assertTrue(BayesianCausalForest.__instancecheck__(m_apply_approach.call_args[0][0]))

        self.assertEqual(m_apply_approach.call_count, len(approaches))

    def test_cast_to_dataframe(self):
        helper = HelperPipeline()

        list_dict = [{
            'A-0': 0.0,
            'A-1': -0.0014,
            'A-2': -0.0063,
            'A-3': 0.0039,
            'A-4': 0.007,
            'A-5': 0.003,
            'A-6': -0.0085,
            'A-7': -0.0156,
            'A-8': -0.0087,
            'A-9': -0.0093,
            'A-10': -0.0149
        }, {
            'B-0': 0.0,
            'B-1': -0.0153,
            'B-2': -0.0029,
            'B-3': -0.0063,
            'B-4': -0.0061,
            'B-5': -0.0051,
            'B-6': -0.0106,
            'B-7': -0.0081,
            'B-8': -0.0094,
            'B-9': -0.0187,
            'B-10': -0.0149
        }, {
            'A-0': 0.0,
            'A-1': -0.008,
            'A-2': -0.0064,
            'A-3': -0.0113,
            'A-4': -0.0219,
            'A-5': -0.0195,
            'A-6': -0.0172,
            'A-7': -0.0083,
            'A-8': -0.0102,
            'A-9': -0.0151,
            'A-10': -0.0149
        }, {
            'B-0': 0.0,
            'B-1': -0.0052,
            'B-2': -0.0093,
            'B-3': -0.0076,
            'B-4': -0.0096,
            'B-5': -0.0065,
            'B-6': -0.0077,
            'B-7': -0.0082,
            'B-8': -0.0131,
            'B-9': -0.015,
            'B-10': -0.0149
        }, {
            'A-0': 0.0,
            'A-1': 0.0016,
            'A-2': -0.018,
            'A-3': -0.0171,
            'A-4': -0.014,
            'A-5': -0.015,
            'A-6': -0.0176,
            'A-7': -0.0131,
            'A-8': -0.0129,
            'A-9': -0.007,
            'A-10': -0.0149
        }, {
            'B-0': 0.0,
            'B-1': 0.0043,
            'B-2': -0.0013,
            'B-3': -0.0076,
            'B-4': -0.0117,
            'B-5': -0.0115,
            'B-6': -0.0163,
            'B-7': -0.0103,
            'B-8': -0.0108,
            'B-9': -0.0114,
            'B-10': -0.0149
        }]

        print("##### LIST DICT #####")
        print(list_dict)
        df_uplift = helper.cast_to_dataframe(list_dict)
        print("##### DATAFRAME #####")
        print(df_uplift)

        # Check if type equals pd.DataFrame
        self.assertEqual(type(df_uplift), pd.DataFrame)

        # Check if the DataFrame contains 55 columns (11 columns for each approach)
        self.assertEqual(df_uplift.shape[1], 22)

    def test_save_feature_importance(self):
        pass
