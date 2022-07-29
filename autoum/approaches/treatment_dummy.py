import logging
import pickle
from datetime import datetime

import pandas as pd
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

        df_train_with_interactions = data_set_helper.df_train.copy()
        df_train_with_interactions.drop(['response'], axis=1, inplace=True)
        if data_set_helper.valid:
            df_valid_with_interactions = data_set_helper.df_valid.copy()
            df_valid_with_interactions.drop(['response'], axis=1, inplace=True)
        df_test_with_interactions = data_set_helper.df_test.copy()
        df_test_with_interactions.drop(['response'], axis=1, inplace=True)

        columns = df_test_with_interactions.drop(['treatment'], axis=1).columns

        # Calculate interaction effects
        for column in columns:
            df_train_with_interactions[column + "_x_treatment"] = df_train_with_interactions[column] * df_train_with_interactions["treatment"]
            df_test_with_interactions[column + "_x_treatment"] = df_test_with_interactions[column] * df_test_with_interactions["treatment"]

            if data_set_helper.valid:
                df_valid_with_interactions[column + "_x_treatment"] = df_valid_with_interactions[column] * df_valid_with_interactions["treatment"]

        if self.cost_sensitive:
            # Calculate class weights
            class_weights = Helper.create_class_weight(data_set_helper.y_train)
            clf = LogisticRegression(class_weight=class_weights)
        else:
            clf = LogisticRegression()

        clf.set_params(**self.parameters)
        self.log.debug("Start fitting Treatment Dummy ...")

        clf.fit(df_train_with_interactions.to_numpy(), data_set_helper.y_train)
        self.log.debug(clf)
        response_dict = {}

        if self.save:
            self.log.debug("Saving ...")
            date_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            filename = self.path + 'results/models/{}_Treatment_Dummy_{}.pickle'.format(str(self.split_number), date_str)
            pickle.dump(clf, open(filename, 'wb'))

        self.log.debug("Predicting ...")

        # Training
        response_dict["score_train"] = TreatmentDummy.prediction(clf, df_train_with_interactions)

        # Validation
        if data_set_helper.valid:
            response_dict["score_valid"] = TreatmentDummy.prediction(clf, df_valid_with_interactions)

        # Test
        response_dict["score_test"] = TreatmentDummy.prediction(clf, df_test_with_interactions)

        return response_dict

    @staticmethod
    def prediction(clf, df_with_interactions: pd.DataFrame):
        """
        Predict the uplift scores for the given data set

        :param clf: Classifier
        :param df_with_interactions: DataFrame which should be analyzed
        :return: Uplift scores
        """

        df_with_interactions["treatment"] = 1
        score_train_t_1 = clf.predict_proba(df_with_interactions.to_numpy())[:, 1]
        df_with_interactions["treatment"] = 0
        score_train_t_0 = clf.predict_proba(df_with_interactions.to_numpy())[:, 1]

        return score_train_t_1 - score_train_t_0
