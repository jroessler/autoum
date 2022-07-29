import logging
import pickle
from datetime import datetime

import numpy as np
from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV

from autoum.approaches.utils import ApproachParameters, DataSetsHelper


# TODO: Cost sensitive learning
# TODO: Feature Importance


class GeneralizedRandomForest:
    """
    Generalized Random Forest proposed by Athey et al. (2019)
    """

    def __init__(self, parameters: dict, approach_parameters: ApproachParameters):
        """
        Creates a classifier for the generalized random forest (Athey et al., 2019)

        :param parameters: The parameters needed for the creation of the base learner
        :param approach_parameters: Pass an approach_parameters object that contains all parameters necessary to execute the approach
        """

        self.parameters = parameters
        self.parameters["model_t"] = LassoCV()
        self.parameters["model_y"] = LassoCV()
        self.feature_importance = approach_parameters.feature_importance
        self.save = approach_parameters.save
        self.path = approach_parameters.path
        self.split_number = approach_parameters.split_number
        self.log = logging.getLogger(type(self).__name__)

    def analyze(self, data_set_helper: DataSetsHelper) -> dict:
        """
        Calculate the score (ITE/Uplift/CATE) for each sample using generalized random forest

        :param data_set_helper: A DataSetsHelper comprising the training, validation (optional) and test data set
        :return: Dictionary containing, scores and feature importance
        """

        # Initiate CausalForestDML
        causal_forrest_classifier = CausalForestDML(**self.parameters)

        treated = data_set_helper.df_train["treatment"].copy()

        self.log.debug("Start fitting Generalized Random Forest ...")
        causal_forrest_classifier.fit(Y=data_set_helper.y_train, T=treated, X=data_set_helper.x_train, W=None)

        self.log.debug(causal_forrest_classifier)

        if self.save:
            self.log.debug("Saving ...")
            date_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            filename = self.path + 'results/models/{}_GRF_{}.pickle'.format(str(self.split_number), date_str)
            pickle.dump(causal_forrest_classifier, open(filename, 'wb'))

        self.log.debug("Predicting ... ")

        # Note: The .predict() method returns an ndarray in which each column contains the predicted uplift if the unit was in the corresponding treatment group.
        score_train = causal_forrest_classifier.effect(data_set_helper.x_train)
        score_test = causal_forrest_classifier.effect(data_set_helper.x_test)
        if data_set_helper.valid:
            score_valid = causal_forrest_classifier.effect(data_set_helper.x_valid)
        else:
            score_valid = np.array([0], np.int32)

        return {
            "score_train": score_train.flatten(),
            "score_valid": score_valid.flatten(),
            "score_test": score_test.flatten()
        }
