import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from sklearn.model_selection import train_test_split

from autouplift.datasets.utils import get_hillstrom_women_visit
from autouplift.evaluation.evaluation import UpliftEvaluation
from autouplift.pipelines.pipeline_rw import PipelineRW
from autouplift.pipelines.utils import HelperPipeline


class TestPipelineRW(unittest.TestCase):

    def setUp(self):
        # Get data
        data = get_hillstrom_women_visit()
        self.data = data.sample(frac=0.5, random_state=123)
        self.df_train, self.df_test = train_test_split(data, test_size=0.2, shuffle=True, stratify=data[['response', 'treatment']], random_state=123)

    @patch('tests.test_pipeline_rw.PipelineRW.analyze_k_fold_cv', spec_set=True)
    @patch('tests.test_pipeline_rw.PipelineRW.analyze_single_fold', spec_set=True)
    @patch('tests.test_pipeline_rw.PipelineRW.calculate_metrics', spec_set=True)
    def test_analyze_dataset(self, m_calculate_metrics, m_analyze_single_fold, m_analyze_k_fold_cv):
        for cv_number_splits in [2, 10]:
            with self.subTest(i=cv_number_splits):
                pipeline = PipelineRW(cv_number_splits=cv_number_splits)
                # Test function
                pipeline.analyze_dataset(self.data)

                if cv_number_splits == 10:
                    m_analyze_k_fold_cv.assert_called_once()
                    self.assertTrue(m_analyze_k_fold_cv.call_args[1]['df_train'].shape[1] == self.data.shape[1])
                    self.assertAlmostEqual(m_analyze_k_fold_cv.call_args[1]['df_train'].shape[0], int(self.data.shape[0] * 0.8), delta=2)
                    self.assertTrue(m_analyze_k_fold_cv.call_args[1]['df_test'].shape[1] == self.data.shape[1])
                    self.assertAlmostEqual(m_analyze_k_fold_cv.call_args[1]['df_test'].shape[0], int(self.data.shape[0] * 0.2), delta=2)
                else:
                    m_analyze_single_fold.assert_called_once()
                    self.assertTrue(m_analyze_single_fold.call_args[1]['df_train'].shape[1] == self.data.shape[1])
                    self.assertAlmostEqual(m_analyze_single_fold.call_args[1]['df_train'].shape[0], int(self.data.shape[0] * 0.8), delta=2)
                    self.assertTrue(m_analyze_single_fold.call_args[1]['df_test'].shape[1] == self.data.shape[1])
                    self.assertAlmostEqual(m_analyze_single_fold.call_args[1]['df_test'].shape[0], int(self.data.shape[0] * 0.2), delta=2)

                m_calculate_metrics.assert_called_once()
                m_calculate_metrics.reset_mock()

    def test_analyze_k_fold(self):
        pass

    @patch('tests.test_pipeline_rw.PipelineRW.train_eval_splits', spec_set=True)
    def test_analyze_single_fold(self, m_train_eval_splits):
        pipeline = PipelineRW(cv_number_splits=2, urf_cts=False)
        approaches = ["BCF", "CVT", "GRF", "LAIS", "RLEARNER", "SLEARNER", "XLEARNER", "TRADITIONAL", "TREATMENT_DUMMY", "TWO_MODEL", "URF_ED", "URF_KL", "URF_CHI", "URF_DDP",
                      "URF_IT", "URF_CIT"]
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

    def test_create_single_split(self):
        pass

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

        # Case 1.1: Approach (S-Learner) and no optimal qini curve
        pipeline = PipelineRW()
        args = (0, self.df_test, self.df_test, self.df_test, 'SLEARNER')
        self.check_function(pipeline, dict_uplift, dict_opt_uplift_empty, args)

        # Case 1.2: Approach (S-Learner) and optimal qini curve
        pipeline = PipelineRW(metrics_qini_coefficient=True)
        args = (0, self.df_test, self.df_test, self.df_test, 'SLEARNER')
        self.check_function(pipeline, dict_uplift, dict_opt_uplift, args, metric_qini_coefficient=True)

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
        # Check if the approach name is equal to the supposed approach name
        self.assertTrue(HelperPipeline.apply_uplift_approaches.call_args[1]['approach'] == [args[4]])
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

    def test_calculate_metrics(self):
        for feature_importance in [True, False]:
            for metrics_qini_coefficient in [True, False]:
                for plot_figures in [True, False]:
                    with self.subTest(i=feature_importance):
                        list_feature_importances = []
                        list_dict_uplift_train = list_dict_uplift_valid = list_dict_uplift_test = [{
                            'Uplift_0': 0.0,
                            'Uplift_1': 0.0041,
                            'Uplift_2': 0.0101,
                            'Uplift_3': 0.02,
                            'Uplift_4': 0.0301,
                            'Uplift_5': 0.0368,
                            'Uplift_6': 0.0391,
                            'Uplift_7': 0.0438,
                            'Uplift_8': 0.044,
                            'Uplift_9': 0.0418,
                            'Uplift_10': 0.0454
                        }]
                        list_dict_opt_uplift_train = list_dict_opt_uplift_valid = list_dict_opt_uplift_test = []
                        feature_names = ["A", "B", "C"]

                        pipeline = PipelineRW(cv_number_splits=2, feature_importance=feature_importance, metrics_qini_coefficient=metrics_qini_coefficient,
                                              plot_figures=plot_figures)
                        PipelineRW.calculate_feature_importance_mean = MagicMock(spec_set=True)
                        HelperPipeline.cast_to_dataframe = MagicMock(return_value=pd.DataFrame, spec_set=True)
                        UpliftEvaluation.calculate_unscaled_qini_coefficient = MagicMock(return_value=pd.DataFrame, spec_set=True)
                        UpliftEvaluation.calculate_qini_coefficient = MagicMock(return_value=pd.DataFrame, spec_set=True)
                        UpliftEvaluation.calculate_mean = MagicMock(return_value=pd.DataFrame, spec_set=True)
                        PipelineRW.plotting = MagicMock(spec_set=True)

                        pipeline.calculate_metrics(list_feature_importances, list_dict_uplift_train, list_dict_uplift_valid, list_dict_uplift_test, list_dict_opt_uplift_train,
                                                   list_dict_opt_uplift_valid, list_dict_opt_uplift_test, feature_names)

                        if feature_importance:
                            self.assertEqual(PipelineRW.calculate_feature_importance_mean.call_count, 1)
                        else:
                            self.assertEqual(PipelineRW.calculate_feature_importance_mean.call_count, 0)
                        self.assertEqual(HelperPipeline.cast_to_dataframe.call_count, 6)
                        self.assertEqual(UpliftEvaluation.calculate_unscaled_qini_coefficient.call_count, 3)
                        if metrics_qini_coefficient:
                            self.assertEqual(UpliftEvaluation.calculate_qini_coefficient.call_count, 3)
                        else:
                            self.assertEqual(UpliftEvaluation.calculate_qini_coefficient.call_count, 0)
                        self.assertEqual(UpliftEvaluation.calculate_mean.call_count, 3)
                        if plot_figures:
                            self.assertEqual(PipelineRW.plotting.call_count, 3)
                        else:
                            self.assertEqual(PipelineRW.plotting.call_count, 0)

    def test_plotting(self):
        pass

    def test_create_approach_tuples(self):
        cv_number_splits = 10
        pipeline = PipelineRW(cv_number_splits=cv_number_splits, urf_ddp=False, two_model=False)
        dataframe_pairs = pipeline.create_k_splits(df_train=self.df_train, df_test=self.df_test)
        tuple_list = pipeline.create_approach_tuples(dataframe_pairs)
        self.assertEqual(len(tuple_list), 15 * cv_number_splits)
        for _tuple in tuple_list:
            self.assertEqual(len(_tuple), 5)

    def test_create_approach_list_for_single_split(self):
        pipeline = PipelineRW(cv_number_splits=2, urf_cts=False)
        all_approaches = pipeline.create_approach_list_for_single_split()
        self.assertEqual(len(all_approaches), 16)


if __name__ == '__main__':
    unittest.main(verbosity=2)
