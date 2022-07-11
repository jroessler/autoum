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

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](https://github.com/jroessler/autouplift/blob/main/LICENSE) file 
for details.

## References

### Documentation

### Citation

To cite AutoUplift in publications, you can refer to the following paper:
TODO add