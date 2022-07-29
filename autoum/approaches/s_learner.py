import logging
import pickle
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier

from autoum.approaches.utils import ApproachParameters, DataSetsHelper, Helper


class SLearner:
    """
    S-Learner proposed in multiple papers (e.g., Künzel et al. 2019)

    Here, the treatment variable is used as another dependent variable during training.
    """

    def __init__(self, parameters: dict, approach_parameters: ApproachParameters):
        """
        Creates a classifier for the S-Learner

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
        Calculate the score (ITE/Uplift/CATE) for each sample using the S-Learner.

        :param data_set_helper: A DataSetsHelper comprising the training, validation (optional) and test data set
        :return: Dictionary containing, scores and feature importance
        """

        if self.cost_sensitive:
            # Calculate class weights
            class_weights = Helper.create_class_weight(data_set_helper.y_train)
            clf = RandomForestClassifier(class_weight=class_weights)
        else:
            clf = RandomForestClassifier()

        clf.set_params(**self.parameters)

        self.log.debug("Start fitting S-Learner ...")

        # Create data frame including the treatment variable but not response variable
        df_train_with_treatment = data_set_helper.df_train.drop(['response'], axis=1)
        df_test_with_treatment = data_set_helper.df_test.drop(['response'], axis=1)

        clf.fit(df_train_with_treatment.to_numpy(), data_set_helper.y_train)

        self.log.debug(clf)

        response_dict = {}

        if self.save:
            self.log.debug("Saving ...")
            date_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            filename = self.path + 'results/models/{}_S-Learner_{}.pickle'.format(str(self.split_number), date_str)
            pickle.dump(clf, open(filename, 'wb'))

        if self.feature_importance:
            response_dict["feature_importance"] = clf.feature_importances_

        self.log.debug("Predicting ...")

        score_dict = {}
        df_train_with_treatment["treatment"] = 0
        score_dict["score_train_t_0"] = clf.predict_proba(df_train_with_treatment.to_numpy())[:, 1]
        df_train_with_treatment["treatment"] = 1
        score_dict["score_train_t_1"] = clf.predict_proba(df_train_with_treatment.to_numpy())[:, 1]
        response_dict["score_train"] = score_dict["score_train_t_1"] - score_dict["score_train_t_0"]

        df_test_with_treatment["treatment"] = 0
        score_dict["score_test_t_0"] = clf.predict_proba(df_test_with_treatment.to_numpy())[:, 1]
        df_test_with_treatment["treatment"] = 1
        score_dict["score_test_t_1"] = clf.predict_proba(df_test_with_treatment.to_numpy())[:, 1]
        response_dict["score_test"] = score_dict["score_test_t_1"] - score_dict["score_test_t_0"]

        if data_set_helper.valid:
            df_valid_with_treatment = data_set_helper.df_valid.drop(['response'], axis=1)
            df_valid_with_treatment["treatment"] = 0
            score_dict["score_valid_t_0"] = clf.predict_proba(df_valid_with_treatment.to_numpy())[:, 1]
            df_valid_with_treatment["treatment"] = 1
            score_dict["score_valid_t_1"] = clf.predict_proba(df_valid_with_treatment.to_numpy())[:, 1]
            response_dict["score_valid"] = score_dict["score_valid_t_1"] - score_dict["score_valid_t_0"]

        else:
            response_dict["score_valid"] = []

        return response_dict
