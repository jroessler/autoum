import logging
import pickle
from datetime import datetime

import numpy as np
from causalml.inference.tree import UpliftRandomForestClassifier

from autoum.approaches.utils import ApproachParameters, DataSetsHelper, Helper


# TODO: Cost sensitive learning
# TODO: Feature Importance


class UpliftRandomForest:
    """
    Uplift Random Forest proposed by Rzepakowski and Jaroszewicz (2012)

    Modify existing supervised learning algorithms to directly infer a causal effect.
    According to the current literature, decision trees and different ensembles of decision trees are the most popular adjusted algorithms.
    Usually, the tree-building algorithm and the splitting criterion are modified such that they maximize the difference in uplift.

    The Uplift Tree approach consists of a set of methods that use a tree-based algorithm (Random Forest) where the
    splitting criterion is based on differences in uplift (N. Rzepakowski, Jaroszewicz, 2012) and (Hansotia & Rukstales, 2002)

    You can choose among the following evaluation functions:
    "ED": Euclidean Distance
    "KL": Kullback-Leibler Divergence
    "Chi" Chi-Square Divergence
    "DDP": Delta Delta P
    "CTS": Contextual Treatment Selection
    "IT": Interaction Tree (Su et al., 2009)
    "CIT": Causal Inference Tree (Su et al., 2012)
    """

    def __init__(self, parameters: dict, approach_parameters: ApproachParameters, eval_function: str):
        """
        Creates a classifier for the uplift random forest

        :param parameters: The parameters needed for the creation of the base learner
        :param approach_parameters: Pass an approach_parameters object that contains all parameters necessary to execute the approach
        :param eval_function: Evaluation function. You can choose among the following evaluation functions:
            "ED": Euclidean Distance
            "KL": Kullback-Leibler Divergence
            "Chi" Chi-Square Divergence
            "DDP": Delta Delta P
            "CTS": Contextual Treatment Selection
            "IT": Interaction Tree (Su et al., 2009)
            "CIT": Causal Inference Tree (Su et al., 2012)
        """

        self.parameters = parameters
        self.parameters["evaluationFunction"] = eval_function
        self.feature_importance = approach_parameters.feature_importance
        self.save = approach_parameters.save
        self.path = approach_parameters.path
        self.split_number = approach_parameters.split_number
        self.log = logging.getLogger(type(self).__name__)

    def analyze(self, data_set_helper: DataSetsHelper) -> dict:
        """
        Calculate the score (ITE/Uplift/CATE) for each sample using uplift random forest

        :param data_set_helper: A DataSetsHelper comprising the training, validation (optional) and test data set
        :return: Dictionary containing, scores and feature importance
        """

        # Add causalML's 'treatment' column
        experiment_groups_col = Helper.add_treatment_group_key(data_set_helper.df_train)

        # Initiate UpliftRandomForestClassifier
        urf = UpliftRandomForestClassifier(**self.parameters)

        self.log.debug("Start fitting Uplift Random Forest ...")

        urf.fit(X=data_set_helper.x_train, treatment=experiment_groups_col, y=data_set_helper.y_train)

        self.log.debug(urf)

        if self.save:
            self.log.debug("Saving ...")
            date_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            filename = self.path + 'results/models/{}_Direct_Uplift_{}.pickle'.format(str(self.split_number), date_str)
            pickle.dump(urf, open(filename, 'wb'))

        self.log.debug("Predicting ... ")

        # Note: The .predict() method returns an ndarray in which each column contains the predicted uplift if the unit was in the corresponding treatment group.
        score_train = urf.predict(data_set_helper.x_train)
        score_test = urf.predict(data_set_helper.x_test)
        if data_set_helper.valid:
            score_valid = urf.predict(data_set_helper.x_valid)
        else:
            score_valid = np.array([0], np.int32)

        return {
            "score_train": score_train.flatten(),
            "score_valid": score_valid.flatten(),
            "score_test": score_test.flatten()
        }
