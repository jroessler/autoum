import logging
import pickle
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from autoum.approaches.utils import ApproachParameters, DataSetsHelper, Helper


class ClassVariableTransformation:
    """
    Class Variable Transformation proposed by Jaskowski & Jaroszewicz (2012)

    A new target variable is created using the following transformation:
    response * treatment + (1 - response) * (1 - treatment)

    Assumption:
    P(T=0) = P(T=1) = 1/2 !!

    The classifier is based on Random Forest algorithm.
    """

    def __init__(self, parameters: dict, approach_parameters: ApproachParameters):
        """
        Creates a classifier for the class variable transformation (Jaskowski & Jaroszewicz, 2012))

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
        Calculate the score (ITE/Uplift/CATE) for each sample using class variable transformation

        :param data_set_helper: A DataSetsHelper comprising the training, validation (optional) and test data set
        :return: Dictionary containing, scores and feature importance
        """

        # Sanity Check: Check if treatment and control group are almost equal (maximum difference being 0.1)
        df_treatment = data_set_helper.df_train['treatment']
        diff = np.abs(df_treatment.value_counts()[1] / df_treatment.shape[0] - df_treatment.value_counts()[0] / df_treatment.shape[0])
        if diff >= 0.1:
            self.log.error(
                f"Assumption P(G=T)=P(G=C)=1/2 for the transformed outcome approach by Jaskowski & Jaroszewicz (2012) is violated with a difference between treatment and"
                f"control group of {diff:.3f}. This approach will be skipped!")
            raise ValueError("Assumption P(G=T)=P(G=C)=1/2 for the transformed outcome approach by Jaskowski & Jaroszewicz (2012) is violated")

        # Transform the response
        y_train = ClassVariableTransformation.transform(data_set_helper.df_train['response'].to_numpy(), data_set_helper.df_train['treatment'].to_numpy())
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            raise Exception("Training set only contains samples from one class. Availabe Class : {}".format(unique_classes))

        if self.cost_sensitive:
            # Calculate class weights
            class_weights = Helper.create_class_weight(y_train)
            clf = RandomForestClassifier(class_weight=class_weights)
        else:
            clf = RandomForestClassifier()

        clf.set_params(**self.parameters)

        self.log.debug("Start fitting Class Variable Transformation ...")

        clf.fit(data_set_helper.x_train, y_train)
        transformed_dict = {}

        if self.feature_importance:
            transformed_dict["feature_importance"] = clf.feature_importances_

        self.log.debug(clf)

        if self.save:
            self.log.debug("Saving ...")
            date_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            filename = self.path + 'results/models/{}_CVT_{}.pickle'.format(str(self.split_number), date_str)
            pickle.dump(clf, open(filename, 'wb'))

        self.log.debug("Predicting ... ")

        transformed_dict["score_train"] = 2 * clf.predict_proba(data_set_helper.x_train)[:, 1] - 1
        transformed_dict["score_test"] = 2 * clf.predict_proba(data_set_helper.x_test)[:, 1] - 1
        if data_set_helper.valid:
            transformed_dict["score_valid"] = 2 * clf.predict_proba(data_set_helper.x_valid)[:, 1] - 1
        else:
            transformed_dict["score_valid"] = []

        return transformed_dict

    @staticmethod
    def transform(y: np.ndarray, treat: np.ndarray) -> np.ndarray:
        """
        Transforms the target variable y with the following transformation

        y_trans = y * treat + (1-y) * (1-treat)

        Assumption: During the campaign, the treatment was randomly assigned to individuals.

        :param y: Target variable to be transformed
        :param treat: Treatment variable
        :return: Transformed target variable
        """

        return y * treat + (1 - y) * (1 - treat)
