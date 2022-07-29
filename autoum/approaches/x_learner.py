import logging
import pickle
from datetime import datetime

import numpy as np
from causalml.inference.meta import BaseXClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from autoum.approaches.utils import ApproachParameters, DataSetsHelper, Helper


# TODO: Cost sensitive learning
# TODO: Feature Importance


class XLearner:
    """
    X-Learner proposed by Künzel et al. (2019)

    X-learner is an extension of T-learner (Two-Model Approach), and consists of three stages as follows:

    Stage 1
    Estimate the response functions using any machine learning algorithm

    𝜇1(𝑥)=𝐸[𝑌(1)|𝑋=𝑥]; 𝜇'1=> Response function for the treatment group "treatment estimator", and
    𝜇0(𝑥)=𝐸[𝑌(0)|𝑋=𝑥]; 𝜇'0=> Response function for the control group "control estimator"

    Stage 2
    Impute the individiual treatment effects:

    𝐷1𝑖=𝑌1𝑖−𝜇̂ 0(𝑋1𝑖); ITE for the treatment individuals is defined as difference between true outcome of the treatment individual minus the outcome of the control estimator, and
    𝐷0𝑖=𝜇̂ 1(𝑋0𝑖)−𝑌0𝑖; ITE for the control individuals is defined as difference between the outcome of the treatment estimator and the true outcome of the control individual

    Example:
        Individuum A was in the treatment group and responded (that is what we have observed):
            * True Outcome: Treatment Responder
            * Outcome of the control estimator: e.g., Control Non-Responder
            * ITE (𝐷1): Treatment Responder - Control Non-Responder (e.g., 1-0 = 1)

    After imputing the ITE, estimate 𝜏1(𝑥)=𝐸[𝐷1|𝑋=𝑥], and 𝜏0(𝑥)=𝐸[𝐷0|𝑋=𝑥] using any machine learning algorithm.
    That is, we will use the imputed treatment effects D1 and D0 as response variable to obtain 𝜏1 and 𝜏0 (CATE)

    Stage 3
    Define the CATE estimate by a weighted average of 𝜏1(𝑥) and 𝜏0(𝑥):

    𝜏(𝑥)=𝑔(𝑥)𝜏0(𝑥)+(1−𝑔(𝑥))𝜏1(𝑥)

    where 𝑔∈[0,1] is a weight funciton. We can use propensity scores for 𝑔(𝑥).
    """

    def __init__(self, parameters: dict, approach_parameters: ApproachParameters):
        """
        Creates a classifier for the  X-learner approach

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
        Calculate the score (ITE/Uplift/CATE) for each sample using the X-Learner.

        Side note: For randomized controlled experiments, we can use the "observed propensity score", that is, the ratio between the individuals who were subject to a treatment
        and the individuals who were not subject to a treatment. For observational data, we need to estimate the propensity score.

        :param data_set_helper: A DataSetsHelper comprising the training, validation (optional) and test data set
        :return: Dictionary containing, scores and feature importance
        """

        # Add causalML's 'treatment' column
        experiment_groups_col = Helper.add_treatment_group_key(data_set_helper.df_train)

        # Initiate outcome and effect learner
        outcome_learner = RandomForestClassifier(**self.parameters)  # treatment and control estimator - stage 1. Should be a classifier
        effect_learner = RandomForestRegressor(**self.parameters)  # effect (CATE, tau) estimator - stage 2. Should be a regressor

        self.log.debug("Start fitting X-Learner ...")

        # Get the covariates for the data set
        x_train = data_set_helper.x_train
        x_test = data_set_helper.x_test

        # X Learner
        num_treatment_samples = np.count_nonzero(data_set_helper.df_train['treatment'].to_numpy() == 1)
        x_learner = BaseXClassifier(outcome_learner=outcome_learner, effect_learner=effect_learner, control_name='c')
        x_learner.fit(X=x_train, treatment=experiment_groups_col, y=data_set_helper.y_train, p=Helper.get_propensity_score(x_train.shape[0], num_treatment_samples))

        self.log.debug(x_learner)

        if self.save:
            self.log.debug("Saving ...")
            date_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            filename = self.path + 'results/models/{}_X-Learner_{}.pickle'.format(str(self.split_number), date_str)
            pickle.dump(x_learner, open(filename, 'wb'))

        self.log.debug("Predicting ...")

        score_train = x_learner.predict(X=x_train, p=Helper.get_propensity_score(x_train.shape[0]))
        score_test = x_learner.predict(X=x_test, p=Helper.get_propensity_score(x_test.shape[0]))

        if data_set_helper.valid:
            x_valid = data_set_helper.x_valid
            score_valid = x_learner.predict(X=x_valid, p=Helper.get_propensity_score(x_valid.shape[0]))
        else:
            score_valid = np.array([0], np.int32)

        return {
            "score_train": score_train.flatten(),
            "score_valid": score_valid.flatten(),
            "score_test": score_test.flatten()
        }
