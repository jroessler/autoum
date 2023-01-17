import os
import logging
import pickle
from datetime import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression

from autoum.approaches.utils import ApproachParameters, DataSetsHelper, Helper


class TreatmentDummy:
    """
    Treatment dummy approach with Logistic Regression proposed by Lo (2002)
    """

    def __init__(self, parameters: dict, approach_parameters: ApproachParameters):
        """
        Creates a classifier for the treatment dummy approach with logistic regression

        :param parameters: The parameters needed for the creation of the base learner
        :param approach_parameters: Pass an approach_parameters object that contains all parameters necessary to execute the approach
        """

        self.parameters = parameters
        self.cost_sensitive = approach_parameters.cost_sensitive
        self.feature_importance = approach_parameters.feature_importance
        self.save = approach_parameters.save
        self.path = approach_parameters.path
        self.split_number = approach_parameters.split_number
        self.log = logging.getLogger(type(self).__name__)

    def analyze(self, data_set_helper: DataSetsHelper) -> dict:
        """
        Calculate the score (ITE/Uplift/CATE) for each sample using the treatment dummy approach.

        :param data_set_helper: A DataSetsHelper comprising the training, validation (optional) and test data set
        :return: Dictionary containing, scores and feature importance
        """

        # Get training's treatments and features
        train_treatment = data_set_helper.df_train['treatment'].to_numpy()
        train_features = data_set_helper.df_train.drop(['treatment', 'response'], axis=1).to_numpy()
        # Get test's features
        test_features = data_set_helper.df_test.drop(['treatment', 'response'], axis=1).to_numpy()

        if self.cost_sensitive:
            # Calculate class weights
            class_weights = Helper.create_class_weight(data_set_helper.y_train)
            clf = LogisticRegression(class_weight=class_weights)
        else:
            clf = LogisticRegression()

        clf.set_params(**self.parameters)
        self.log.debug("Start fitting Treatment Dummy ...")

        # Train classifier with features * interaction effects
        clf.fit(np.append(np.append(train_features, train_features * train_treatment[:, None], axis=1), train_treatment[:, None], axis=1), data_set_helper.y_train)
        self.log.debug(clf)
        response_dict = {}

        if self.save:
            self.log.debug("Saving ...")
            date_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            path = self.path + 'results/models/'
            filename = path + f'{self.split_number}_TDA_{date_str}.pickle'
            if not os.path.exists(path):
                os.makedirs(path)
            pickle.dump(clf, open(filename, 'wb'))

        self.log.debug("Predicting ...")

        # Training
        response_dict["score_train"] = TreatmentDummy.prediction(clf, train_features)

        # Validation
        if data_set_helper.valid:
            # Get validation's features
            valid_features = data_set_helper.df_valid.drop(['treatment', 'response'], axis=1).to_numpy()
            response_dict["score_valid"] = TreatmentDummy.prediction(clf, valid_features)

        # Test
        response_dict["score_test"] = TreatmentDummy.prediction(clf, test_features)

        return response_dict

    @staticmethod
    def prediction(clf, features: np.array):
        """
        Predict the uplift scores for the given data set

        :param clf: Classifier
        :param features: Numpy array containing the features
        :return: Uplift scores
        """

        # Create an array with features + interaction effects + treatment (==1)
        score_train_t_1 = clf.predict_proba(np.append(np.append(features, features * np.ones((features.shape[0], 1)), axis=1), np.ones((features.shape[0], 1)), axis=1))[:, 1]
        # Create an array with features + interaction effects (are all zero because of the treatment) + treatment (==0)
        score_train_t_0 = clf.predict_proba(np.append(np.append(features, features * np.zeros((features.shape[0], 1)), axis=1), np.zeros((features.shape[0], 1)), axis=1))[:, 1]

        return score_train_t_1 - score_train_t_0
