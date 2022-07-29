import logging
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from autoum.approaches.utils import ApproachParameters, DataSetsHelper, Helper


class LaisGeneralization:
    """
    Lais Generalization proposed by Kane et al. (2014)

    A new target variable is created using an arbitrary transformation.

    Assumption (without Generalization):
    P(T=1) = P(T=1) = 1/2 !!

    Thanks to Kane (2015) we can cope with any P(T) and P(C) this is often refered to as "Lais Generalization"
    """

    def __init__(self, parameters: dict, approach_parameters: ApproachParameters):
        """
        Creates a classifier for Lai's (Lais 2015) generalization (Kane et al, 2015)

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
        Calculate the score (ITE/Uplift/CATE) for each sample using Lai's generalization

        :param data_set_helper: A DataSetsHelper comprising the training, validation (optional) and test data set
        :return: Dictionary containing, scores and feature importance
        """

        # Transform the response
        y_train = LaisGeneralization.transform(data_set_helper.df_train)

        unique_classes = np.unique(y_train)
        if len(unique_classes) < 4:
            raise Exception("Training set only contains samples from three or less class. Available Classses {}".format(unique_classes))

        if self.cost_sensitive:
            # Calculate class weights
            class_weights = Helper.create_class_weight(y_train)
            clf = RandomForestClassifier(class_weight=class_weights)
        else:
            clf = RandomForestClassifier()

        clf.set_params(**self.parameters)

        self.log.debug("Start fitting Lai's Generalization ...")

        clf.fit(data_set_helper.x_train, y_train)

        self.log.debug(clf)
        transformed_dict = {}

        if self.save:
            self.log.debug("Saving ...")
            date_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            filename = self.path + 'results/models/{}_LG_{}.pickle'.format(str(self.split_number), date_str)
            pickle.dump(clf, open(filename, 'wb'))

        if self.feature_importance:
            transformed_dict["feature_importance"] = clf.feature_importances_

        self.log.debug("Predicting ... ")

        df_train = data_set_helper.df_train
        df_test = data_set_helper.df_test

        prob_t_train = df_train.loc[df_train.treatment == 1].shape[0] / df_train.shape[0]
        prob_t_test = df_test.loc[df_test.treatment == 1].shape[0] / df_test.shape[0]

        x_train = data_set_helper.x_train
        x_test = data_set_helper.x_test

        # (P(TR) / P(T)) + (P(CN) / P(C)) - (P(TN) / P(T)) - (P(CR) / P(C))
        transformed_dict["score_train"] = (clf.predict_proba(x_train)[:, 3] / prob_t_train) + (clf.predict_proba(x_train)[:, 0] / 1 - prob_t_train) - (
                clf.predict_proba(x_train)[:, 1] / prob_t_train) - (clf.predict_proba(x_train)[:, 2] / 1 - prob_t_train)
        transformed_dict["score_test"] = (clf.predict_proba(x_test)[:, 3] / prob_t_test) + (clf.predict_proba(x_test)[:, 0] / 1 - prob_t_test) - (
                clf.predict_proba(x_test)[:, 1] / prob_t_test) - (clf.predict_proba(x_test)[:, 2] / 1 - prob_t_test)

        if data_set_helper.valid:
            df_valid = data_set_helper.df_valid
            prob_t_valid = df_valid.loc[df_valid.treatment == 1].shape[0] / df_valid.shape[0]
            x_valid = data_set_helper.x_valid
            transformed_dict["score_valid"] = (clf.predict_proba(x_valid)[:, 3] / prob_t_valid) + (clf.predict_proba(x_valid)[:, 0] / 1 - prob_t_valid) - (
                    clf.predict_proba(x_valid)[:, 1] / prob_t_valid) - (clf.predict_proba(x_valid)[:, 2] / 1 - prob_t_valid)
        else:
            transformed_dict["score_valid"] = []

        return transformed_dict

    @staticmethod
    def transform(_df_train: pd.DataFrame) -> np.ndarray:
        """
        Transforms the target variable y with the following transformation

        T = 0 & R = 0 --> 1
        T = 1 & R = 0 --> 2
        T = 0 & R = 1 --> 3
        T = 1 & R = 1 --> 4

        Assumption: During the campaign, the treatment was randomly assigned to individuals.

        :param _df_train: Trainings DataFrame
        :return: Transformed target variable
        """

        df = _df_train.copy()
        df.loc[((df['treatment'] == 0) & (df['response'] == 0)), 'group'] = 1
        df.loc[((df['treatment'] == 1) & (df['response'] == 0)), 'group'] = 2
        df.loc[((df['treatment'] == 0) & (df['response'] == 1)), 'group'] = 3
        df.loc[((df['treatment'] == 1) & (df['response'] == 1)), 'group'] = 4

        return df["group"].to_numpy()
