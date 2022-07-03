import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
root = os.getenv("ROOT_FOLDER")
sys.path.append(root + "src/")

from pipelines.helper.helper_pipeline import HelperPipeline
from evaluation.evaluation import UpliftEvaluation

# This is the class we want to test in this file
from pipelines.pipeline_rw import PipelineRW


class TestPipelineRW(unittest.TestCase):

    def setUp(self):

        # Helper
        self.helper = HelperPipeline()

        # Dataset
        self.dataset_name = "Companye_k"

        # Trainings & Test DataFrame
        self.df_train, self.df_test = self.helper.get_dataframe(dataset_name=self.dataset_name, test_size=0.2, random_seed=123)

    @patch('test_pipeline_rw.PipelineRW.analyze_k_fold_cv', spec_set=True)
    @patch('test_pipeline_rw.PipelineRW.analyze_single_fold', spec_set=True)
    @patch('test_pipeline_rw.PipelineRW.calculate_metrics', spec_set=True)
    def test_analyze_dataset(self, m_calculate_metrics, m_analyze_single_fold, m_analyze_k_fold_cv):
        for cv_number_splits in [2, 10]:
            with self.subTest(i=cv_number_splits):
                pipeline = PipelineRW(cv_number_splits=cv_number_splits)
                # Test function
                pipeline.analyze_dataset(self.dataset_name)

                if cv_number_splits == 10:
                    m_analyze_k_fold_cv.assert_called_once()
                    self.assertTrue(m_analyze_single_fold.call_args[1]['df_train'].equals(self.df_train))
                    self.assertTrue(m_analyze_single_fold.call_args[1]['df_test'].equals(self.df_test))
                else:
                    m_analyze_single_fold.assert_called_once()
                    self.assertTrue(m_analyze_single_fold.call_args[1]['df_train'].equals(self.df_train))
                    self.assertTrue(m_analyze_single_fold.call_args[1]['df_test'].equals(self.df_test))

                m_calculate_metrics.assert_called_once()
                m_calculate_metrics.reset_mock()

    @patch('test_pipeline_rw.PipelineRW.train_eval_splits', spec_set=True)
    def test_analyze_single_fold(self, m_train_eval_splits):
        pipeline = PipelineRW(cv_number_splits=2)
        approaches = ['TWO_MODEL', 'JASKO_JARO']
        PipelineRW.create_approach_list_for_single_split = MagicMock(return_value=approaches, spec_set=True)
        # m_create_approach_list_for_single_split.return_value = approaches
        m_train_eval_splits.return_value = ({}, {}, {}, {}, {}, {}, {})

        # Test function
        dict_list_dict_uplift = pipeline.analyze_single_fold(self.df_train, self.df_test)

        # Check if create_approach_list_for_single_split was called once
        PipelineRW.create_approach_list_for_single_split.assert_called_once()
        # Check if train_eval_splits was called len(approaches) times
        self.assertEqual(m_train_eval_splits.call_count, len(approaches))

        # Check if we have receveied one list for each approach
        self.assertEqual(len(dict_list_dict_uplift['list_dict_uplift_train']), len(approaches))
        self.assertEqual(len(dict_list_dict_uplift['list_dict_uplift_valid']), len(approaches))
        self.assertEqual(len(dict_list_dict_uplift['list_dict_uplift_test']), len(approaches))
        self.assertEqual(len(dict_list_dict_uplift['list_dict_opt_uplift_train']), len(approaches))
        self.assertEqual(len(dict_list_dict_uplift['list_dict_opt_uplift_valid']), len(approaches))
        self.assertEqual(len(dict_list_dict_uplift['list_dict_opt_uplift_test']), len(approaches))
        self.assertEqual(len(dict_list_dict_uplift['feature_importances']), len(approaches))

    def test_create_k_splits(self):
        for cv_number_splits in [5, 10]:
            with self.subTest(i=cv_number_splits):
                pipeline = PipelineRW(cv_number_splits=cv_number_splits)
                dataframe_pairs = pipeline.create_k_splits(self.df_train, self.df_test)

                self.assertEqual(len(dataframe_pairs), cv_number_splits)
                for dataframe_pair in dataframe_pairs:
                    self.assertEqual(len(dataframe_pair), 4)
                    self.assertIsInstance(dataframe_pair[0], int)
                    self.assertIsInstance(dataframe_pair[1], pd.DataFrame)
                    self.assertIsInstance(dataframe_pair[2], pd.DataFrame)
                    self.assertIsInstance(dataframe_pair[3], pd.DataFrame)

    def test_train_eval_splits(self):
        # Mocking apply_uplift_approaches
        scores_dict = {
            "df_scores_train": pd.DataFrame(),
            "df_scores_valid": pd.DataFrame(),
            "df_scores_test": pd.DataFrame(),
            "feature_importances": {}
        }
        HelperPipeline.apply_uplift_approaches = MagicMock(return_value=scores_dict, spec_set=True)

        # Mocking calculate_actual_uplift_in_bins
        dict_uplift = {
            'Uplift_0': 0.0000,
            'Uplift_1': 0.0831,
            'Uplift_2': 0.1288,
            'Uplift_3': 0.1154,
            'Uplift_4': 0.1021,
            'Uplift_5': 0.0900,
            'Uplift_6': 0.0788,
            'Uplift_7': 0.0664,
            'Uplift_8': 0.0611,
            'Uplift_9': 0.0470,
            'Uplift_10': 0.0454
            }
        UpliftEvaluation.calculate_actual_uplift_in_bins = MagicMock(return_value=dict_uplift, spec_set=True)

        # Mocking calculate_optimal_uplift_in_bins
        UpliftEvaluation.calculate_optimal_uplift_in_bins = MagicMock(spec_set=True)
        dict_opt_uplift_empty = {}
        dict_opt_uplift = {
            'Uplift_0': 0.0000,
            'Uplift_1': 0.0831,
            'Uplift_2': 0.1288,
            'Uplift_3': 0.1154,
            'Uplift_4': 0.1021,
            'Uplift_5': 0.0900,
            'Uplift_6': 0.0788,
            'Uplift_7': 0.0664,
            'Uplift_8': 0.0611,
            'Uplift_9': 0.0470,
            'Uplift_10': 0.0454
        }
        UpliftEvaluation.calculate_optimal_uplift_in_bins.side_effect = [dict_opt_uplift, dict_opt_uplift, dict_opt_uplift, dict_opt_uplift, dict_opt_uplift, dict_opt_uplift]

        # Case 1.1: Direct approach and no optimal qini curve
        pipeline = PipelineRW()
        metric_qini_coefficient = False
        args = (0, self.df_test, self.df_test, self.df_test, 'DIRECT', metric_qini_coefficient, 'DDP')
        self.check_function(pipeline, dict_uplift, dict_opt_uplift_empty, args)

        # Case 1.2: Direct approach and optimal qini curve
        pipeline = PipelineRW(metrics_qini_coefficient=True)
        metric_qini_coefficient = True
        args = (0, self.df_test, self.df_test, self.df_test, 'DIRECT', metric_qini_coefficient, 'DDP')
        self.check_function(pipeline, dict_uplift, dict_opt_uplift, args, metric_qini_coefficient=metric_qini_coefficient)

        # Case 2.1: Not direct approach and no optimal qini curve
        pipeline = PipelineRW()
        metric_qini_coefficient = False
        args = (0, self.df_test, self.df_test, self.df_test, 'TRADITIONAL', metric_qini_coefficient)
        self.check_function(pipeline, dict_uplift, dict_opt_uplift_empty, args)

        # Case 2.2: Not direct approach and optimal qini curve
        pipeline = PipelineRW(metrics_qini_coefficient=True)
        metric_qini_coefficient = True
        args = (0, self.df_test, self.df_test, self.df_test, 'TRADITIONAL', metric_qini_coefficient)
        self.check_function(pipeline, dict_uplift, dict_opt_uplift, args, metric_qini_coefficient=metric_qini_coefficient)

    def check_function(self, pipeline, dict_uplift, dict_opt_uplift, args, metric_qini_coefficient=False):
        """
        Check different train_eval_splits calls

        :param pipeline: PipelineRW
        :param dict_uplift: Dictionary containing the uplift values for each bin/decile and approach
        :param dict_opt_uplift: Dictionary containing the optimal plift values for each bin/decile and approach
        :param args: Arguments for the train_eval_splits call
        :param metric_qini_coefficient: True if the qini coefficient should be calculated. False otherwise.
        """
        *output_train_eval, = pipeline.train_eval_splits(args)
        # Check if each output is of instance dict
        self.assertIsInstance(output_train_eval[0], dict)
        self.assertIsInstance(output_train_eval[1], dict)
        self.assertIsInstance(output_train_eval[2], dict)
        self.assertIsInstance(output_train_eval[3], dict)
        self.assertIsInstance(output_train_eval[4], dict)
        self.assertIsInstance(output_train_eval[5], dict)
        self.assertIsInstance(output_train_eval[6], dict)
        # Check length of each output
        self.assertEqual(len(output_train_eval[0]), len(dict_uplift))
        self.assertEqual(len(output_train_eval[1]), len(dict_uplift))
        self.assertEqual(len(output_train_eval[2]), len(dict_uplift))
        self.assertEqual(len(output_train_eval[3]), len(dict_opt_uplift))
        self.assertEqual(len(output_train_eval[4]), len(dict_opt_uplift))
        self.assertEqual(len(output_train_eval[5]), len(dict_opt_uplift))

        # Check if apply_uplift_approaches was called once
        HelperPipeline.apply_uplift_approaches.assert_called_once()
        # Check if calculate_actual_uplift_in_bins was called 3 times (for training, validation, and testing)
        self.assertEqual(UpliftEvaluation.calculate_actual_uplift_in_bins.call_count, 3)
        # Check if calculate_optimal_uplift_in_bins was called 3 times (if metric_qini_coefficient == True). 0 times otherwise.
        if metric_qini_coefficient:
            self.assertEqual(UpliftEvaluation.calculate_optimal_uplift_in_bins.call_count, 3)
        else:
            UpliftEvaluation.calculate_optimal_uplift_in_bins.assert_not_called()
        HelperPipeline.apply_uplift_approaches.reset_mock()
        UpliftEvaluation.calculate_actual_uplift_in_bins.reset_mock()
        UpliftEvaluation.calculate_optimal_uplift_in_bins.reset_mock()


if __name__ == '__main__':
    unittest.main(verbosity=2)
