import unittest
from unittest.mock import patch

from autoum.pipelines.pipeline_sd import PipelineSD


class TestPipelineSD(unittest.TestCase):

    def setUp(self):
        self.n = 50000
        self.p = 20
        self.sigma = 1.0
        self.propensity = 0.5
        self.threshold = 50

    @patch('tests.test_pipeline_sd.PipelineSD.analyze_k_fold_cv', spec_set=True)
    @patch('tests.test_pipeline_sd.PipelineSD.analyze_single_fold', spec_set=True)
    @patch('tests.test_pipeline_sd.PipelineSD.calculate_metrics', spec_set=True)
    def test_analyze_dataset(self, m_calculate_metrics, m_analyze_single_fold, m_analyze_k_fold_cv):
        for cv_number_splits in [2, 10]:
            with self.subTest(i=cv_number_splits):
                pipeline = PipelineSD(self.n, self.p, self.sigma, self.threshold, self.propensity, cv_number_splits=cv_number_splits)
                # Test function
                pipeline.analyze_dataset()

                if cv_number_splits == 10:
                    m_analyze_k_fold_cv.assert_called_once()
                else:
                    m_analyze_single_fold.assert_called_once()

                m_calculate_metrics.assert_called_once()
                m_calculate_metrics.reset_mock()

    def test_create_synthetic_dataset(self):
        pipeline = PipelineSD(self.n, self.p, self.sigma, self.threshold, self.propensity)
        df = pipeline.create_synthetic_dataset()

        self.assertEqual(df.shape[0], self.n)
        self.assertEqual(df.shape[1], self.p + 2)
        self.assertEqual(round(df.treatment.value_counts()[1] / df.shape[0], 1), self.propensity)
