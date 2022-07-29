import logging
import pickle
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from autoum.approaches.utils import ApproachParameters, DataSetsHelper, Helper


class Traditional:
    """
    Traditional Response Modeling Approach

    Here, only the treatment group is used for training the classifier
    """

    def __init__(self, parameters: dict, approach_parameters: ApproachParameters):
        """
        Creates a classifier for the traditional approach

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
        Calculate the score (ITE/Uplift/CATE) for each sample using the traditional approach.

        :param data_set_helper: A DataSetsHelper comprising the training, validation (optional) and test data set
        :return: Dictionary containing, scores and feature importance
        """

        # Training (Build dataset which contains only the treated individuals)
        df_train_treat = data_set_helper.df_train.loc[data_set_helper.df_train.treatment == 1]  # DataFrame
        x_train_treat = df_train_treat.drop(['response', 'treatment'], axis=1).to_numpy()  # Features
        y_train_treat = df_train_treat['response'].to_numpy()  # Target

        # Validation (Features)
        if data_set_helper.valid:
            x_valid = data_set_helper.df_valid.drop(['response', 'treatment'], axis=1).to_numpy()
        else:
            x_valid = None

        # Testing (Features)
        x_test = data_set_helper.df_test.drop(['response', 'treatment'], axis=1).to_numpy()
        # Training (for prediciton purposes we need the whole data set)
        x_train = data_set_helper.df_train.drop(['response', 'treatment'], axis=1).to_numpy()

        unique_classes = np.unique(y_train_treat)
        if len(unique_classes) < 2:
            self.log.error(f"Training set only contains samples from one class. Available Classes: {unique_classes}")
            raise Exception("Training set only contains samples from one class. Available Classes: {}".format(unique_classes))

        if self.cost_sensitive:
            # Calculate class weights
            class_weights = Helper.create_class_weight(y_train_treat)
            clf = RandomForestClassifier(class_weight=class_weights)
        else:
            clf = RandomForestClassifier()

        clf.set_params(**self.parameters)

        self.log.debug("Start fitting Traditional ...")

        clf.fit(x_train_treat, y_train_treat)

        self.log.debug(clf)

        if self.save:
            self.log.debug("Saving ...")
            date_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            filename = self.path + 'results/models/{}_Tradtional_{}.pickle'.format(str(self.split_number), date_str)
            pickle.dump(clf, open(filename, 'wb'))

        traditional_dict = {}

        if self.feature_importance:
            traditional_dict["feature_importance"] = clf.feature_importances_

        self.log.debug("Predicting ...")

        traditional_dict["score_train"] = clf.predict_proba(x_train)[:, 1]
        traditional_dict["score_test"] = clf.predict_proba(x_test)[:, 1]
        if x_valid is not None:
            traditional_dict["score_valid"] = clf.predict_proba(x_valid)[:, 1]
        else:
            traditional_dict["score_valid"] = []

        return traditional_dict
