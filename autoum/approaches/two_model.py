import logging
import pickle
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier

from autoum.approaches.utils import DataSetsHelper, Helper


class TwoModel:
    """
    Two Model Approach proposed in multiple papers (e.g., Künzel et al. 2019)

    Also called T-Learner
    """

    def __init__(self, parameters: dict, approach_parameters):
        """
        Creates a classifier for the two model approach

        :param parameters: The parameters needed for the creation of the base learner
        :param approach_parameters: Pass an ApproachParameters object that contains all parameters necessary to execute the approach
        """
        self.parameters = parameters
        self.cost_sensitive = approach_parameters.cost_sensitive
        self.feature_importance = approach_parameters.feature_importance
        self.save = approach_parameters.save
        self.path = approach_parameters.path
        self.split_number = approach_parameters.split_number
        self.log = logging.getLogger(type(self).__name__)

    def training(self, x, y, group: str):
        """
        Train & create a classifier

        :param x: Features
        :param y: Targets
        :param group: Name of the model. Either 'Treatment' or 'Control'
        :return: Model
        """
        self.log.debug(f"Start fitting {group} random forest")

        if self.cost_sensitive:
            # Calculate class weights
            # TODO: Raise Exception if len(class_weights) <2 ? Yes
            class_weights = Helper.create_class_weight(y)
            self.log.debug("Class weights: " + str(class_weights))

            clf = RandomForestClassifier(class_weight=class_weights)
        else:
            clf = RandomForestClassifier()

        clf.set_params(**self.parameters)
        clf.fit(x, y)
        self.log.debug(clf)

        return clf

    def save_model(self, clf, group: str):
        """
        Save the given model as pickle file.

        :param clf: Model
        :param group: Name of the model. Either 'Treatment' or 'Control'
        """
        self.log.debug(f"Saving {group} Model...")
        date_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        filename = self.path + f'results/models/{str(self.split_number)}_Two_Model_{group}_{date_str}.pickle'
        pickle.dump(clf, open(filename, 'wb'))

    @staticmethod
    def prediction(clf_treat, clf_non_treat, x):
        """
        Predict the probabilities for responding using the Treatment and the Control model

        :param clf_treat: Treatment model
        :param clf_non_treat: Control model
        :param x: Features
        :return: score_train_treat (probabilities from Treatment model), score_train_non_treat (probabilities from Control model)
        """
        score_train_treat = clf_treat.predict_proba(x)
        score_train_non_treat = clf_non_treat.predict_proba(x)

        return score_train_treat, score_train_non_treat

    def validate_results(self, prob_treat, prob_non_treat, sample: str):
        """
        Sanity check for the results. Check if the groups (i.e., Treatment responder, Treatment non responder, control responder, and control non responder) are well represented.

        :param prob_treat: Probabilities from the Treatment model
        :param prob_non_treat: Probabilities from the Control model
        :param sample: String referring to the current sample (i.e., Training, Validation, or Test)
        :return: Uplift Scores
        """
        # TODO: Check whether assert / exception in case we only have only treatment or control groups is necessary
        if prob_treat.shape[1] < 2:
            self.log.debug(f"{sample}: No samples with treated = 1 and response = 1")
            uplift_score = 0 - prob_non_treat[:, 1]
        elif prob_non_treat.shape[1] < 2:
            self.log.debug(f"{sample}: No samples with treated = 0 and response = 1")
            uplift_score = prob_treat[:, 1] - 0
        else:
            uplift_score = prob_treat[:, 1] - prob_non_treat[:, 1]

        return uplift_score

    def analyze(self, data_set_helper: DataSetsHelper) -> dict:
        """
        Calculate the score (ITE/Uplift/CATE) for each sample using the TwoModelRFClassifier.

        :param data_set_helper: A DataSetsHelper comprising the training, validation (optional) and test data set
        :return: Dictionary containing, scores and feature importance
        """

        # Build two different datasets, one including the treated individuals and one including the individuals not treated
        df_train_treat = data_set_helper.df_train.loc[data_set_helper.df_train.treatment == 1]
        df_train_non_treat = data_set_helper.df_train.loc[data_set_helper.df_train.treatment == 0]

        # Target (for training)
        y_train_treat = df_train_treat['response'].to_numpy()
        y_train_non_treat = df_train_non_treat['response'].to_numpy()

        # Features (for training)
        x_train_treat = df_train_treat.drop(['response', 'treatment'], axis=1).to_numpy()
        x_train_non_treat = df_train_non_treat.drop(['response', 'treatment'], axis=1).to_numpy()

        # Training
        clf_treat = self.training(x_train_treat, y_train_treat, "Treatment")
        clf_non_treat = self.training(x_train_non_treat, y_train_non_treat, "Control")

        if self.save:
            self.save_model(clf_treat, "Treatment")
            self.save_model(clf_non_treat, "Control")

        two_model_dict = {}

        # Feature importance
        if self.feature_importance:
            two_model_dict["feature_importance"] = {
                "feature_importance_treated": clf_treat.feature_importances_,
                "feature_importance_untreated": clf_non_treat.feature_importances_
            }

        # Predicting
        self.log.debug('Predicting ...')

        # Prediciton on Training
        score_train_treat, score_train_non_treat = TwoModel.prediction(clf_treat, clf_non_treat, data_set_helper.x_train)
        uplift_score_train = self.validate_results(score_train_treat, score_train_non_treat, "Training")
        two_model_dict["score_train"] = uplift_score_train

        # Prediction on Validation
        if data_set_helper.valid:
            score_valid_treat, score_valid_non_treat = TwoModel.prediction(clf_treat, clf_non_treat, data_set_helper.x_valid)
            uplift_score_valid = self.validate_results(score_valid_treat, score_valid_non_treat, "Validation")
            two_model_dict["score_valid"] = uplift_score_valid
        else:
            two_model_dict["score_valid"] = []

        # Prediction on Test
        score_test_treat, score_test_non_treat = TwoModel.prediction(clf_treat, clf_non_treat, data_set_helper.x_test)
        uplift_score_test = self.validate_results(score_test_treat, score_test_non_treat, "Test")
        two_model_dict["score_test"] = uplift_score_test

        return two_model_dict
