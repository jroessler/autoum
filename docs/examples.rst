Examples
========

Working example notebooks are available in the `example folder <https://github.com/jroessler/autoum/tree/main/examples>`_.

Pipeline for Real-World Datasets
--------------------------------

.. code-block:: python

    from autoum.datasets.utils import get_hillstrom_women_visit
    from autoum.pipelines.pipeline_rw import PipelineRW

    data = get_hillstrom_women_visit()
    pipeline = PipelineRW(
        bayesian_causal_forest=True,
        cv_number_splits=10,
        class_variable_transformation=True,
        generalized_random_forest=True,
        lais_generalization=True,
        max_depth=5,
        min_samples_leaf=50,
        min_samples_treatment=10,
        n_estimators=20,
        plot_figures=True,
        plot_uqc=True,
        rlearner=True,
        run_name="AutoUM",
        run_id=1,
        slearner=True,
        show_title=True,
        traditional=True,
        treatment_dummy=True,
        two_model=True,
        urf_ed=True,
        urf_kl=True,
        urf_chi=True,
        urf_ddp=True,
        urf_cts=True,
        urf_it=True,
        urf_cit=True,
        xlearner=True
    )

    pipeline.analyze_dataset(data)

.. image:: ./_static/img/Real_World_Qini_Curve.png
    :width: 630

Pipeline for Synthetic Datasets
-------------------------------

.. code-block:: python

    from autoum.pipelines.pipeline_sd import PipelineSD

    n_samples = 20000  # 20.000 samples / rows
    n_covariates = 20  # 20 covariates / columns
    sigma = 0.5  # Covariance of 0.5
    treatment_propensity = 0.5  # treatment propensity of 0.5 (i.e. 50:50)
    response_rate = 20  # 20% response rate

    pipeline = PipelineSD(
        n=n_samples,
        p=n_covariates,
        sigma=sigma,
        threshold=response_rate,
        propensity=treatment_propensity,
        cv_number_splits=5,
        generalized_random_forest=True,
        max_depth=5,
        min_samples_leaf=50,
        min_samples_treatment=10,
        n_estimators=20,
        plot_figures=True,
        plot_uqc=True,
        run_name="Synthetic_Example",
        show_title=True,
        traditional=True,
        two_model=True,
        urf_ed=True
    )

    data = pipeline.create_synthetic_dataset()
    pipeline.analyze_dataset(data)

.. image:: ./_static/img/Synthetic_Real_World_Qini_Curve.png
    :width: 630