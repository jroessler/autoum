from autouplift.datasets.utils import get_hillstrom_women_visit
from autouplift.pipelines.pipeline_rw import PipelineRW

if __name__ == '__main__':
    data = get_hillstrom_women_visit()
    print(data.head())
    # pipeline = PipelineRW(bayesian_causal_forest=True, cv_number_splits=2, class_variable_transformation=False, logging_level=2, generalized_random_forest=False,
    #                       lais_generalization=True, max_depth=5, metrics_calculate_absolute=True, metrics_save_metrics=True, min_samples_leaf=50, min_samples_treatment=10,
    #                       n_estimators=20, plot_figures=True, plot_uqc=True, plot_save_figures=True, rlearner=False, run_name="Example", run_id=2, slearner=True, show_title=True,
    #                       traditional=False, treatment_dummy=False, two_model=False, urf_ed=False, urf_kl=False, urf_chi=False, urf_ddp=True, urf_cts=False, urf_it=False,
    #                       urf_cit=False, xlearner=True)
    # pipeline.analyze_dataset(data)

    pipeline = PipelineRW(feature_importance=True, lais_generalization=True, max_depth=5, metrics_calculate_absolute=True, metrics_save_metrics=True, min_samples_leaf=50,
                          min_samples_treatment=10, n_estimators=20, plot_figures=True, plot_uqc=True, plot_save_figures=True, run_name="Example", run_id=3, show_title=True,
                          two_model=True)
    pipeline.analyze_dataset(data)
