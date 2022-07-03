import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
root = os.getenv("ROOT_FOLDER")
sys.path.append(root + "src/")

# This is the class we want to test in this file
from evaluation.evaluation import UpliftEvaluation


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        np.random.seed(123)

        n = 10000
        self.treatment = np.random.binomial(n=1, p=0.5, size=[n])
        self.response = np.random.binomial(n=1, p=0.5, size=[n])
        self.uplift_score = np.random.normal(0.3, 0.1, n)

        self.df = pd.DataFrame(data={'treatment': self.treatment, 'response': self.response, 'uplift_score': self.uplift_score})

    def test_separate_treated_control(self):
        df_treated, df_control = UpliftEvaluation.separate_treated_control(df_results=self.df, treatment_col='treatment', response_col='response', uplift_score_col='uplift_score')
        uplift_treated = df_treated.uplift_score.tolist()
        uplift_control = df_control.uplift_score.tolist()

        # Check if values are sorted descending by uplift score value
        self.assertTrue(all(uplift_treated[i] >= uplift_treated[i + 1] for i in range(len(uplift_treated) - 1)))
        self.assertTrue(all(uplift_control[i] >= uplift_control[i + 1] for i in range(len(uplift_control) - 1)))

        # Check if the length of treatment and control group is correct
        self.assertTrue(df_treated.shape[0], np.count_nonzero(df_treated.uplift_score))
        self.assertTrue(df_control.shape[0], np.count_nonzero(df_control.uplift_score))

    def test_calculate_uplift(self):
        bin_treatment_responder = [70, 140, 210, 280, 350, 420, 490, 560, 630]
        bin_number_treated_samples = [100, 200, 300, 400, 500, 600, 700, 800, 900]
        bin_non_treatment_responder = [30, 60, 90, 120, 150, 180, 210, 240, 270]
        bin_number_non_treated_samples = [100, 200, 300, 400, 500, 600, 700, 800, 900]
        bins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        num_treated_samples = 4500

        expected_uplift_abs = [0., 40., 80., 120., 160., 200., 240., 280., 320., 360.]
        expected_uplift_pct = [np.around(uplift / num_treated_samples, 4) for uplift in expected_uplift_abs]

        bin_list, uplift_abs, uplift_pct = UpliftEvaluation.calculate_uplift(bin_treatment_responder, bin_number_treated_samples, bin_non_treatment_responder,
                                                                             bin_number_non_treated_samples, bins, num_treated_samples)

        # Check if the size of each list (bin_list uplift & uplift_pct) is equal to bin + 1
        self.assertEqual(len(bin_list), len(bins) + 1)
        self.assertEqual(len(uplift_abs), len(bins) + 1)
        self.assertEqual(len(uplift_pct), len(bins) + 1)

        # Check if uplift was calculated correct
        self.assertListEqual(uplift_abs.tolist(), expected_uplift_abs)
        self.assertListEqual(uplift_pct.tolist(), expected_uplift_pct)

    def test_calculate_qini_curve(self):
        tretment_responder = self.df.loc[(self.df['treatment'] == 1) & (self.df['response'] == 1)].shape[0]
        control_responder = self.df.loc[(self.df['treatment'] == 0) & (self.df['response'] == 1)].shape[0]

        control_samples = self.df.loc[(self.df['treatment'] == 0)].shape[0]
        treated_samples = self.df.loc[(self.df['treatment'] == 1)].shape[0]

        treatment_response_rate = tretment_responder / treated_samples
        control_response_rate = control_responder / control_samples

        abs_endpoint = treatment_response_rate - control_response_rate

        for col in self.df.drop(['response', 'treatment'], axis=1).columns:
            for bin_size in [10, 100]:
                bin_list, uplift, uplift_pct = UpliftEvaluation.calculate_qini_curve(self.df, uplift_score_col=col, bins=bin_size)
                # Check if the size of each list (bin_list uplift & uplift_pct) is equal to bin + 1
                self.assertEqual(len(bin_list), bin_size + 1)
                self.assertEqual(len(uplift), bin_size + 1)
                self.assertEqual(len(uplift_pct), bin_size + 1)

                # Check if the bin_list is sorted ascending
                self.assertSequenceEqual(bin_list.tolist(), sorted(bin_list))
                # Check if the endpoint calculated for the uplift curve is correct
                self.assertAlmostEqual(uplift_pct[-1], abs_endpoint, places=4)
                # Check if the startpoint calculated for the uplift curve is zero
                self.assertAlmostEqual(uplift_pct[0], 0)

    def test_calculate_optimal_qini_curve(self):
        tretment_responder = self.df.loc[(self.df['treatment'] == 1) & (self.df['response'] == 1)].shape[0]
        control_responder = self.df.loc[(self.df['treatment'] == 0) & (self.df['response'] == 1)].shape[0]

        control_samples = self.df.loc[(self.df['treatment'] == 0)].shape[0]
        treated_samples = self.df.loc[(self.df['treatment'] == 1)].shape[0]

        treatment_response_rate = tretment_responder / treated_samples
        control_response_rate = control_responder / control_samples

        abs_endpoint = treatment_response_rate - control_response_rate

        def check_slope(x):
            """
            Check if slope is first increasing and than decreasing
            """
            # Find maximum
            i = x.argmax()
            increase = x[0:i]
            decrease = x[i:-1]
            return all(x < y for x, y in zip(increase, increase[1:])) and all(x >= y for x, y in zip(decrease, decrease[1:]))

        for bin_size in range(10, 100, 10):
            bin_list, uplift, uplift_pct = UpliftEvaluation.calculate_optimal_qini_curve(self.df, bins=bin_size)

            # Check if the size of each list (bin_list uplift & uplift_pct) is equal to bin + 1
            self.assertEqual(len(bin_list), bin_size + 1)
            self.assertEqual(len(uplift), bin_size + 1)
            self.assertEqual(len(uplift_pct), bin_size + 1)
            # Check if the bin_list is sorted ascending
            self.assertSequenceEqual(bin_list.tolist(), sorted(bin_list))
            # Check if the endpoint calculated for the uplift curve is correct
            self.assertAlmostEqual(uplift_pct[-1], abs_endpoint, places=4)
            # Check if the startpoint is zero
            self.assertAlmostEqual(uplift_pct[0], 0)
            # Check if slope first increases to the maximum and decreases after it (that is a typicall characteristic of an optimal curve)
            self.assertTrue(check_slope(uplift))

    @patch('evaluation.evaluation.UpliftEvaluation.store_uplift_in_bins')
    @patch('evaluation.evaluation.UpliftEvaluation.calculate_qini_curve')
    def test_calculate_actual_uplift_in_bins(self, m_calculate_qini_curve, m_store_uplift_in_bins):
        m_calculate_qini_curve.return_value = ([], [], [])
        uplift_in_deciles = UpliftEvaluation.calculate_actual_uplift_in_bins(self.df, bins=10)

        m_calculate_qini_curve.assert_called_once()
        m_store_uplift_in_bins.assert_called_once()

    @patch('evaluation.evaluation.UpliftEvaluation.store_uplift_in_bins')
    @patch('evaluation.evaluation.UpliftEvaluation.calculate_optimal_qini_curve')
    def test_calculate_optimal_uplift_in_bins(self, m_calculate_optimal_qini_curve, m_store_uplift_in_bins):

        m_calculate_optimal_qini_curve.return_value = ([], [], [])
        uplift_in_deciles = UpliftEvaluation.calculate_optimal_uplift_in_bins(self.df, bins=10)

        UpliftEvaluation.calculate_optimal_qini_curve.assert_called_once()
        UpliftEvaluation.store_uplift_in_bins.assert_called_once()

    def test_calculate_qini_coefficient(self):

        uplift_bins = [0., 1., 3., 5., 7., 9., 10., 10., 10., 10., 10.]
        opt_uplift_bins = [10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.]
        expected_qini_coefficient = 0.4

        df_metrics = pd.DataFrame(data={'test': uplift_bins}).T.reset_index(drop=True)
        df_metrics.columns = ["TT-0", "TT-1", "TT-2", "TT-3", "TT-4", "TT-5", "TT-6", "TT-7", "TT-8", "TT-9", "TT-10"]

        df_opt_uplift_bins = pd.DataFrame(data={'test': opt_uplift_bins}).T.reset_index(drop=True)
        df_opt_uplift_bins.columns = ["TT-0", "TT-1", "TT-2", "TT-3", "TT-4", "TT-5", "TT-6", "TT-7", "TT-8", "TT-9", "TT-10"]

        qini_coefficients_df = UpliftEvaluation.calculate_qini_coefficient(df_metrics, df_opt_uplift_bins, num_columns=11)
        self.assertEqual(expected_qini_coefficient, qini_coefficients_df["TT-QC"].values[0])

        # Check if the shape of the DataFrame has been extened by one column (qini coefficient metric)
        self.assertEqual(df_metrics.shape[1] + 1, qini_coefficients_df.shape[1])
        # Check if the result does not contain any nan values
        self.assertFalse(np.isnan(qini_coefficients_df.iloc[0, -1]))

    def test_calculate_unscaled_qini_coefficient(self):

        uplift_bins = [10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.]
        expected_unscaled_qini_coefficient = 2.0

        df_uplift_bins = pd.DataFrame(data={'test': uplift_bins}).T.reset_index(drop=True)
        df_uplift_bins.columns = ["TT-0", "TT-1", "TT-2", "TT-3", "TT-4", "TT-5", "TT-6", "TT-7", "TT-8", "TT-9", "TT-10"]

        qini_coefficients_df = UpliftEvaluation.calculate_unscaled_qini_coefficient(df_uplift_bins)
        self.assertEqual(expected_unscaled_qini_coefficient, qini_coefficients_df["TT-UQC"].values[0])

        # Check if the shape of the DataFrame has been extened by one column (qini coefficient metric)
        self.assertEqual(df_uplift_bins.shape[1] + 1, qini_coefficients_df.shape[1])
        # Check if the result does not contain any nan values
        self.assertFalse(np.isnan(qini_coefficients_df.iloc[0, -1]))

    def test_calculate_mean(self):
        uplift = np.random.normal(0, 0.01, size=(10, 11))
        uqc = np.random.normal(3, 0.1, size=(10, 1))

        df = pd.DataFrame(data=uplift)
        df.columns = ["TT-0", "TT-1", "TT-2", "TT-3", "TT-4", "TT-5", "TT-6", "TT-7", "TT-8", "TT-9", "TT-10"]
        df["TT-UQC"] = uqc

        df_metrics_mean = UpliftEvaluation.calculate_mean(df)
        self.assertTrue(df_metrics_mean.shape[0] == 1)
