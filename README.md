# AutoUplift: A Python Framework for Automatically Evaluating various Uplift Modeling Algorithms to Estimate Individual Treatment Effects

This Python framework can be used to evaluate multiple methods that predict the potential benefit of a treatment at an individual level. It
provides an interface that allows to estimate the Uplift (also called Conditional Average Treatment Effect (CATE), or Individual Treatment
Effect (ITE)) from **experimental data** with a **binary treatment** indiciator (customers are either treated or not) and a **binary response**
variable (customers either respond or not).

Typical use cases include:

* (Proactive) churn prevention
* Up-/Cross-selling
* Customer acquisition

## Algorithms and Packages

The framework currently supports the following approaches:

* "General" Two Model Approach (e.g., Künzel et al. 2019)
* X-Learner (Künzel et al. 2019)
* Lai's Generalization (Kane et al. 2014)
* Class Variable Transformation (Jaskowski and Jaroszewicz 2012)
* Delta-Delta Pi (DDP) (Hansotia & Rukstales 2002)
* Uplift Random Forest with KL-, ED-, CHI- based splitting criterion (Rzepakowski and Jaroszewicz 2012)
* Contextual Treatment Selection (Zhao et al. 2017)
* Interaction Tree (Su et al. 2009)
* Conditional Interaction Tree (Su et al. 2012)
* Generalized Random Forest (Athey et al. 2019)
* S-Learner (e.g., Künzel et al. 2019)
* Treatment Dummy Approach (Lo 2002)
* Bayesian Causal Forest (Hahn et al. 2020)
* R-Learner (Nie and Wager 2020)
* Traditional Approach (without control group)

There are different statistical approaches to approximate the ITE and several packages in R and Python that implement these. This framework
is mainly build on top of the `CausalML` - Framework.

| Language | Package    | Modeling Approaches                               | Web                                  |
|----------|------------|---------------------------------------------------|--------------------------------------|
| Python   | Causal ML  | Direct Uplift Modeling (tree-based, meta-learner) | https://causalml.readthedocs.io      |
| Python   | CausalLift | Two Model Approach                                | https://github.com/Minyus/causallift |
| Python   | Pylift     | Transformed Outcome (Athey & Imbens 2015)         | https://github.com/wayfair/pylift    |
| Python   | DoWhy      |                                                   | https://github.com/Microsoft/dowhy   |
| Python   | EconML     | Causal Inference                                  | https://github.com/microsoft/EconML  |

## Datasets

The framework contains seven real, publicly available datasets which can be used "on the fly" (see further information below).

| dataset             | Type    | Confidentiallity | Source                                                                                           | Treatment                       | Outcome                          |
|---------------------|---------|------------------|--------------------------------------------------------------------------------------------------|---------------------------------|----------------------------------|
| criteo-marketing    | A/B     | public           | https://s3.us-east-2.amazonaws.com/criteo-uplift-dataset/criteo-uplift.csv.gz                    | `treatment`                     | `conversion`                     |
| hillstrom-email     | A/B     | public           | https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html                | `segment`                       | `spend`, `conversion` or `visit` |
| starbucks           | A/B     | public           | https://github.com/joshxinjie/Data_Scientist_Nanodegree/tree/master/starbucks_portfolio_exercise | `Promotion`                     | `Purchase`                       |
| social-pressure     | A/B     | public           | https://isps.yale.edu/research/data/d001                                                         | `treatment` _(neighbors)_       | `voted`                          |
| lenta               | A/B     | public           | https://www.uplift-modeling.com/en/latest/api/datasets/fetch_lenta.html                          | `group`                         | `response_att`                   |
| criteo-marketing_v2 | A/B     | public           | http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz                                          | `treatment`                     | `conversion`                     |
| bank-telemarketing  | not A/B | public           | https://archive.ics.uci.edu/ml/datasets/bank+marketing                                           | `contact`, `contact`*`poutcome` | `y`                              |


## Installation

So far, only the installation with `pip` is availabe. In the future the installation might be available with `conda`.
Further, please make sure that you are using the latest version of `pip` and `setuptools`.

### Python

Python Version: >=3.8.10

### Install using `pip`
```
$ pip install autouplift
```

For MacOS it might be necessary to run `brew install libomp`

### Install using source

```
$ git clone https://github.com/jroessler/autouplift.git
$ cd autouplift
$ pip install .
```

## Quick Start

### Pipeline for Real-World Datasets

```python
from autouplift.datasets.utils import get_hillstrom_women_visit
from autouplift.pipelines.pipeline_rw import PipelineRW

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
    run_name="AutoUplift",
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
```

See the [Real-World Pipeline Notebook](https://github.com/jroessler/autouplift/blob/main/examples/pipeline_with_real_world_data.ipynb) 
for details.

### Pipeline for Synthetic Datasets

```python
from autouplift.pipelines.pipeline_sd import PipelineSD

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
```
See the [Synthetic Pipeline Notebook](https://github.com/jroessler/autouplift/blob/main/examples/pipeline_with_synthetic_data.ipynb.ipynb) 
for details.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](https://github.com/jroessler/autouplift/blob/main/LICENSE) file 
for details.

## References

### Documentation

### Citation

To cite AutoUplift in publications, you can refer to the following paper:
TODO add