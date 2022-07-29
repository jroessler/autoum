import logging
import pickle
from datetime import datetime

import numpy as np
from xbcausalforest import XBCF

from autoum.approaches.utils import ApproachParameters, DataSetsHelper


# TODO: Cost sensitive learning
# TODO: Feature Importance


class BayesianCausalForest:
    """
    Bayesian Causal Forest (BCF) algorithm proposed by Hahn et al. (2020)

    This algorithm is built on Bayesian additive regression tree (BART)
    An important insight is that the specification of the causal model is a sum of two parts:
    E[Y=y | X=x, T=t]= μ(x) + τ(x)t

    The first part μ(x) is a BART model to estimate the expected outcome from a set of covariates and, if required, an estimate of the probability to receive treatment.
    The parameters for this BART model are denoted with pr for ‘prognostic’, e.g. num_trees_pr.
    The second part τ(x) is a BART model that estimates the treatment effect conditional on some covariates, with its parameters denoted by trt as in ‘treatment’,
    e.g., num_trees_trt
    """

    def __init__(self, parameters: dict, approach_parameters: ApproachParameters):
        """
        Creates a classifier for the Bayesian Causal Forest (Hahn et al, 2020)

        :param parameters: The parameters needed for the creation of the base learner
        :param approach_parameters: Pass an approach_parameters object that contains all parameters necessary to execute the approach
        """

        self.parameters = parameters
        self.save = approach_parameters.save
        self.path = approach_parameters.path
        self.split_number = approach_parameters.split_number
        self.log = logging.getLogger(type(self).__name__)

    def analyze(self, data_set_helper: DataSetsHelper) -> dict:
        """
        Calculate the score (ITE/Uplift/CATE) for each sample using the Bayesian causal forest.

        :param data_set_helper: A DataSetsHelper comprising the training, validation (optional) and test data set
        :return: Dictionary containing, scores and feature importance
        """

        # Prior on the variance in the leaf for each tree. Hahn et al. propose to scale the prior with some factor of the variance of the outcome divided by the number of trees.
        # Default values from here: https://johaupt.github.io/bart/bayesian/causal%20inference/xbcf.html
        self.parameters["tau_pr"] = 0.6 * np.var(data_set_helper.y_train) / self.parameters["num_trees_pr"]
        self.parameters["tau_trt"] = 0.1 * np.var(data_set_helper.y_train) / self.parameters["num_trees_trt"]

        # Count categorical columns
        categorical_cols = 0
        for col in data_set_helper.df_train.drop(["treatment", "response"], axis=1).columns:
            if len(data_set_helper.df_train[col].value_counts()) <= 2:
                categorical_cols += 1
        self.parameters["p_categorical_pr"] = categorical_cols
        self.parameters["p_categorical_trt"] = categorical_cols

        bcf_classifier = XBCF(**self.parameters)

        self.log.debug("Start fitting BayesianCausalForest ...")

        bcf_classifier.fit(x_t=data_set_helper.x_train, x=data_set_helper.x_train, y=data_set_helper.y_train, z=data_set_helper.df_train.treatment)

        self.log.debug(bcf_classifier)

        if self.save:
            self.log.debug("Saving ...")
            date_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            filename = self.path + 'results/models/{}_BCF_{}.pickle'.format(str(self.split_number), date_str)
            pickle.dump(bcf_classifier, open(filename, 'wb'))

        self.log.debug("Predicting ... ")

        # Note: The .predict() method returns a ndarray with the treatment effect for each sample (the higher the value, the more likely the individual is to accept if treated)
        score_train = bcf_classifier.predict(data_set_helper.x_train)
        score_test = bcf_classifier.predict(data_set_helper.x_test)
        if data_set_helper.valid:
            score_valid = bcf_classifier.predict(data_set_helper.x_valid)
        else:
            score_valid = np.array([0], np.int32)

        return {
            "score_train": score_train,
            "score_valid": score_valid,
            "score_test": score_test
        }
