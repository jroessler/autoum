import logging
import pickle
from datetime import datetime

import numpy as np
from causalml.inference.meta import BaseRClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from autoum.approaches.utils import ApproachParameters, DataSetsHelper, Helper


# TODO: Cost sensitive learning
# TODO: Feature Importance


class RLearner:
    """
    R-Learner proposed by Nie and Wager (2020)

    R-Learner is a two-step algorithm for heterogeneous treatment effect estimation.

    1. Step:
    Estimate main effect m*(x) (marginal effect) and treatment propensity e*(x) using (cross-validation) and any machine learning algorithm to form the "R-loss"

    * m*(x): (Informally, in my own words): Use all variables, except for the dependent treatment variable T, to predict Y
    * e*(x): (Informally, in my own words): Use all variables, except for the dependent response variable Y, to predict T (find confounders)

    2. Step:
    Optimize/Minimize the R-Loss Function using m* and e* using any machine learning algorithm
    """

    def __init__(self, parameters: dict, approach_parameters: ApproachParameters):
        """
        Creates a classifier for the R-Learner approach
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
        Calculate the score (ITE/Uplift/CATE) for each sample using the R-Learner.

        Side note: For randomized controlled experiments, we can use the "observed propensity score", that is, the ratio between the individuals who were subject to a treatment
        and the individuals who were not subject to a treatment. For observational data, we need to estimate the propensity score.

        :param data_set_helper: A DataSetsHelper comprising the training, validation (optional) and test data set
        :return: Dictionary containing, scores and feature importance
        """

        # Add causalML's 'treatment' column
        experiment_groups_col = Helper.add_treatment_group_key(data_set_helper.df_train)

        # Initiate outcome and effect learner
        outcome_learner = RandomForestClassifier(**self.parameters)  # estimator for main effect and treatment propensity - stage 1. Should be a classifier
        effect_learner = RandomForestRegressor(**self.parameters)  # estimate the treatment effect - stage 2. Should be a regressor

        self.log.debug("Start fitting R-Learner ...")

        # Get the covariates for the data set
        x_train = data_set_helper.x_train
        x_test = data_set_helper.x_test

        # R Learner
        num_treatment_samples = np.count_nonzero(data_set_helper.df_train['treatment'].to_numpy() == 1)
        r_learner = BaseRClassifier(outcome_learner=outcome_learner, effect_learner=effect_learner, control_name='c', random_state=self.parameters["random_state"])
        r_learner.fit(X=x_train, treatment=experiment_groups_col, y=data_set_helper.y_train, p=Helper.get_propensity_score(x_train.shape[0], num_treatment_samples))

        self.log.debug(r_learner)

        if self.save:
            self.log.debug("Saving ...")
            date_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            filename = self.path + 'results/models/{}_R-Learner_{}.pickle'.format(str(self.split_number), date_str)
            pickle.dump(r_learner, open(filename, 'wb'))

        self.log.debug("Predicting ... ")

        score_train = r_learner.predict(X=x_train)
        score_test = r_learner.predict(X=x_test)

        if data_set_helper.valid:
            x_valid = data_set_helper.x_valid
            score_valid = r_learner.predict(X=x_valid)
        else:
            score_valid = np.array([0], np.int32)

        return {
            "score_train": score_train.flatten(),
            "score_valid": score_valid.flatten(),
            "score_test": score_test.flatten()
        }
