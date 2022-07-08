import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from autouplift.pipelines.pipeline_rw import PipelineRW
from autouplift.pipelines.pipeline_sd import PipelineSD
from autouplift.datasets.utils import get_hillstrom_women_visit

import datetime


class Main:
    """
    Main Class. Executes a Pipeline (either real-world (RW) or synthetic data (SD))
    """

    @staticmethod
    def grid_search(data, **kwargs):
        """
        Grid Search for real-world pipeline
        """

        counter = 0
        for _max_depth in range(25, 51, 25):
            for _n_estimators in range(100, 201, 100):
                for _min_samples_leaf in range(30, 71, 20):
                    kwargs['max_depth'] = _max_depth
                    kwargs['n_estimators'] = _n_estimators
                    kwargs['min_samples_leaf'] = _min_samples_leaf
                    kwargs["run_id"] = counter
                    pipeline = PipelineRW(**kwargs)

                    pipeline.analyze_dataset(data)

                    counter += 1

    @staticmethod
    def single_application(data, **kwargs):
        """
        Single application for real-world pipelineD
        """
        pipeline = PipelineRW(**kwargs)

        pipeline.analyze_dataset(data)

    @staticmethod
    def single_application_synthetic(**kwargs):
        """
        Single application for synthetic pipeline
        :return:
        """
        n = 50000  # --> Nie & Wager 500 and 1000
        p = 20  # -- > Nie & Wager: 6 and 12
        sigmas = [0.5, 1.0, 2.0]
        propensities = [0.2, 0.5, 0.8]
        thresholds = [20, 50, 80]
        counter = 0
        for sigma in sigmas:
            for propensity in propensities:
                for threshold in thresholds:
                    kwargs["run_id"] = counter
                    pipeline = PipelineSD(n, p, sigma, threshold, propensity, **kwargs)
                    pipeline.analyze_dataset()
                    counter += 1


if __name__ == '__main__':
    # TODO
    data = get_hillstrom_women_visit()

    # Get date (for saving purposes)
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')

    parameters = {
        "bayesian_causal_forest": False,
        "bins": 10,
        "cost_sensitive": False,
        "cv_number_splits": 2,
        "class_variable_transformation": False,
        "feature_importance": False,
        "fontsize": 14,
        "generalized_random_forest": False,
        "honesty": False,
        "lais_generalization": False,
        "logging_level": 1,
        "max_depth": 5,
        "max_features": 'auto',
        "metrics_calculate_absolute": False,
        "metrics_qini_coefficient": False,
        "metrics_save_metrics": True,
        "min_samples_leaf": 25,
        "min_samples_treatment": 25,
        "n_estimators": 20,
        "n_jobs_": 1,
        "n_reg": 100,
        "plot_figures": True,
        "plot_optimum": False,
        "plot_grayscale": False,
        "plot_uqc": True,
        "plot_save_figures": False,
        "pool_capacity": 40,
        "rlearner": False,
        "run_name": f"{date}_RUN",
        "run_id": 1,
        "random_seed": 123,
        "save_models": False,
        "slearner": False,
        "show_title": True,
        "test_size": 0.2,
        "traditional": False,
        "treatment_dummy": False,
        "two_model": False,
        "urf_ed": False,
        "urf_kl": False,
        "urf_chi": False,
        "urf_ddp": False,
        "urf_cts": False,
        "urf_it": False,
        "urf_cit": False,
        "validation_size": 0.2,
        "xlearner": True
    }

    # Single Application
    Main.single_application(data=data, **parameters)

    # Grid Application  # Main.grid_search(datasets=_datasets, **parameters)

    # Synthetic Applicaiton  # Main.single_application_synthetic(**parameters)

    # Multi Headed Application  # Main.single_application_multi_headed(datasets=_datasets, **parameters)  # Summary of grid search application for a specific data set  # hillstrom_hon_run_name = "16-11-2021_RUN_"  # hillstrom_hon_data_set = "Hillstrom_Women"  # hillstrom_hon_run_approaches_dict = {  #     "1": ["DDP", "S", "T"],  #     "2": ["ED"],  #     "3": ["R", "X"],  # }

    # Aggregation.join_results(hillstrom_hon_run_name, hillstrom_hon_data_set, hillstrom_hon_run_approaches_dict, number_of_splits=5, bins=10, metrics_qini_coefficient=False)

    # Summary of the results from synthetic application  # Aggregation.aggregate_uplifts_synthetic(run_name="16-11-2021_RUN_", data_set_name="Synthetic", metrics_qini_coefficient=False, number_of_splits=5, bins=10)
