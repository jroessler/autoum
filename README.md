# AutoUM: A Python Framework for Automatically Evaluating various Uplift Modeling Algorithms to Estimate Individual Treatment Effects

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

* "General" Two Model Approach [[1]](#Literature)
* X-Learner (Künzel et al. 2019) [[1]](#Literature)
* Lai's Generalization (Kane et al. 2014) [[2]](#Literature)
* Class Variable Transformation (Jaskowski and Jaroszewicz 2012) [[3]](#Literature)
* Delta-Delta Pi (DDP) (Hansotia & Rukstales 2002) [[4]](#Literature)
* Uplift Random Forest with KL-, ED-, CHI- based splitting criterion (Rzepakowski and Jaroszewicz 2012) [[5]](#Literature)
* Contextual Treatment Selection (Zhao et al. 2017) [[6]](#Literature)
* Interaction Tree (Su et al. 2009) [[7]](#Literature)
* Conditional Interaction Tree (Su et al. 2012) [[8]](#Literature)
* Generalized Random Forest (Athey et al. 2019) [[9]](#Literature)
* S-Learner (e.g., Künzel et al. 2019) [[1]](#Literature)
* Treatment Dummy Approach (Lo 2002) [[10]](#Literature)
* Bayesian Causal Forest (Hahn et al. 2020) [[11]](#Literature)
* R-Learner (Nie and Wager 2020) [[12]](#Literature)
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

### Python

Python Version: >=3.8.10

### Install using `pip`
```
$ pip install autoum
```



### Install using source

```
$ git clone https://github.com/jroessler/autoum.git
$ cd autoum
$ pip install .
```

### Troubleshooting

* Please make sure to keep `pip` and `setuptools` up-to-date
* AutoUM was only tested with MacOS and Linux
* For MacOS it might be necessary to run `brew install libomp`
* Try running the installation with `pip --no-cache-dir install`

## Quick Start

### Pipeline for Real-World Datasets

```python
from autoum.datasets.utils import get_hillstrom_women_visit
from autoum.pipelines.pipeline_rw import PipelineRW

data = get_hillstrom_women_visit()
pipeline = PipelineRW(bayesian_causal_forest=True, cv_number_splits=10, class_variable_transformation=True, generalized_random_forest=True,
    lais_generalization=True, max_depth=5, min_samples_leaf=50, min_samples_treatment=10, n_estimators=20, plot_figures=True, plot_uqc=True,
    rlearner=True, run_name="AutoUM", run_id=1, slearner=True, show_title=True, traditional=True, treatment_dummy=True, two_model=True,
    urf_ed=True, urf_kl=True, urf_chi=True, urf_ddp=True, urf_cts=True, urf_it=True, urf_cit=True, xlearner=True)

pipeline.analyze_dataset(data)
```

See the [Real-World Pipeline Notebook](https://github.com/jroessler/autoum/blob/main/examples/pipeline_with_real_world_data.ipynb) 
for details.

### Pipeline for Synthetic Datasets

```python
from autoum.pipelines.pipeline_sd import PipelineSD

n_samples = 20000  # 20.000 samples / rows
n_covariates = 20  # 20 covariates / columns
sigma = 0.5  # Covariance of 0.5
treatment_propensity = 0.5  # treatment propensity of 0.5 (i.e. 50:50)
response_rate = 20  # 20% response rate

pipeline = PipelineSD(n=n_samples, p=n_covariates, sigma=sigma, threshold=response_rate, propensity=treatment_propensity,
    cv_number_splits=5, generalized_random_forest=True, max_depth=5, min_samples_leaf=50, min_samples_treatment=10, n_estimators=20,
    plot_figures=True, plot_uqc=True, run_name="Synthetic_Example", show_title=True, traditional=True, two_model=True, urf_ed=True)

data = pipeline.create_synthetic_dataset()
pipeline.analyze_dataset(data)
```
See the [Synthetic Pipeline Notebook](https://github.com/jroessler/autoum/blob/main/examples/pipeline_with_synthetic_data.ipynb.ipynb) 
for details.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](https://github.com/jroessler/autoum/blob/main/LICENSE) file 
for details.

## References

### Documentation
[AutoUM API documentation](https://autoum.readthedocs.io/en/latest/index.html)

### Literature

1. Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the national academy of sciences, 116(10), 4156-4165.
2. Kane, K., Lo, V. S., & Zheng, J. (2014). Mining for the truly responsive customers and prospects using true-lift modeling: Comparison of new and existing methods. Journal of Marketing Analytics, 2(4), 218-238.
3. Jaskowski, M., & Jaroszewicz, S. (2012, June). Uplift modeling for clinical trial data. In ICML Workshop on Clinical Data Analysis (Vol. 46, pp. 79-95).
4. Hansotia, B., & Rukstales, B. (2002). Incremental value modeling. Journal of Interactive Marketing, 16(3), 35-46.
5. Rzepakowski, P., & Jaroszewicz, S. (2012). Decision trees for uplift modeling with single and multiple treatments. Knowledge and Information Systems, 32(2), 303-327.
6. Zhao, Y., Fang, X., & Simchi-Levi, D. (2017, June). Uplift modeling with multiple treatments and general response types. In Proceedings of the 2017 SIAM International Conference on Data Mining (pp. 588-596). Society for Industrial and Applied Mathematics.
7. Su, X., Tsai, C. L., Wang, H., Nickerson, D. M., & Li, B. (2009). Subgroup analysis via recursive partitioning. Journal of Machine Learning Research, 10(2).
8. Su, X., Kang, J., Fan, J., Levine, R. A., & Yan, X. (2012). Facilitating score and causal inference trees for large observational studies. Journal of Machine Learning Research, 13, 2955.
9. Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. The Annals of Statistics, 47(2), 1148-1178.
10. Lo, V. S. (2002). The true lift model: a novel data mining approach to response modeling in database marketing. ACM SIGKDD Explorations Newsletter, 4(2), 78-86.
11. Hahn, P. R., Murray, J. S., & Carvalho, C. M. (2020). Bayesian regression tree models for causal inference: Regularization, confounding, and heterogeneous effects (with discussion). Bayesian Analysis, 15(3), 965-1056.
12. Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. Biometrika, 108(2), 299-319.

### More about Uplift Modeling

[Uplift Modeling — An Explanation of the Unknown Challenger in Marketing Campaigns](https://medium.com/@jroessl/uplift-modeling-an-explanation-of-the-unknown-challenger-in-marketing-campaigns-146993613947)