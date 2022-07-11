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


## Policy Regarding Data Files

 * Raw data files should only be committed in compressed form AND using Git LFS https://git-lfs.github.com.
 * Hence, it is necessary to install GIT LFS https://git-lfs.github.com before cloning this repo.
 * After having cloned the repo, compressed data files need to be extracted locally.
 * Usually, uncompressed data files should not be committed (i.e., should be in `.gitignore`; `.csv` already is).
 
## Installation

# For MAC: brew install libomp
 
### Git
 
 ```
$ git clone https://github.com/jroessler/autouplift.git
```

### Python
Python Version: 3.8.10

### Virtual Environment
Create a new virtual environment and install the necessary packages
```
$ cd autouplift
$ python3 -m venv autouplift_env
$ source autouplift_env/bin/activate
$ pip config set global.disable-pip-version-check true
$ pip --no-cache-dir install -r requirements.txt 
$ pip --no-cache-dir install econml==0.12.0
$ pip --no-cache-dir install xbcausalforest==0.1.3
$ cd ..
$ git clone -b v0.12.3 https://github.com/uber/causalml.git
$ cp autouplift/src/approaches/causalml_src/uplift.pyx causalml/causalml/inference/tree/
$ cd causalml
$ python setup.py build_ext --inplace
$ python setup.py install
$ cd ..
$ rm -rf causalml
$ cd autouplift
```
### Development Environment
Create `.env` file (for more information see https://pypi.org/project/python-dotenv/) in root directory with the following content: 
```
ROOT_FOLDER=$PATH$
```
where $PATH$ is the path to your root directory (e.g., ROOT_FOLDER=/home/jroessler/autouplift/)

### Data
* Execute `src/preparation/preparation_main.py` in order to unzip, pre-process and create all publicly available datasets.

## Get started

You can use various pipelines to evaluate different algorithms

* `pipeline_rw`: This pipeline evaluates various uplift modeling approaches on the real-world datasets<br>
* `pipeline_sd`: This pipeline evaluates various uplift modeling approaches on synthetic datasets<br>

All of these Pipelines should be executed via the `src/main/Main.py` class.

Global variables, constants, and similar can be configured in  `src/const/const.py`.

## Own Data

To use your own datasets, do the following steps:
1. Create a new folder in `data/` and place the new dataset inside the folder
2. Define the dataset name and its path in `src/pipelines/helper/helper_pipeline`'s `get_dataframe` method
3. In `src/main.py`, set `_datasets` equal to your name defined in step 2