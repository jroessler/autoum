import numpy as np
import pandas as pd
from sklearn.utils import class_weight


class Helper:
    """
    Helper class for approaches
    """

    @staticmethod
    def create_class_weight(y):
        """
        Create class weights for each class

        :param y: Target column
        :return: A dictionary containing the weights for each class
        """
        unique_classes = np.unique(y)
        class_weights = class_weight.compute_class_weight('balanced', classes=unique_classes, y=y)
        d_class_weights = {}
        for i in range(len(unique_classes)):
            d_class_weights[unique_classes[i]] = class_weights[i]

        return d_class_weights

    @staticmethod
    def add_treatment_group_key(df: pd.DataFrame) -> np.array:
        """
        CausalML expects a "treatment" column which contains a unique string value for individuals belonging to the control group and a unique string value for individuals
        belonging to the treatment group.

        :param df: Dataframe that shall be reformatted
        :return: Array containing String values for the control and the treatment group
        """

        experiment_groups_col = ["c" if x == 0 else "t" for x in df.treatment]
        experiment_groups_col = np.array(experiment_groups_col)

        return experiment_groups_col

    @staticmethod
    def get_propensity_score(length: int, num_treatment_samples: int = 0):
        """
        Return propensity score.
        The propensity score is defined as the probability of receiving a treatment (at least in randomized, controlled experiments)

        :param length: Length of the array
        :param num_treatment_samples: Number of treatment samples in the training set. If None, return 0.5 as propensity score (test set)
        :return: Propensity Score for each
        """

        if num_treatment_samples > 0:
            propensity_score = np.empty(length)
            propensity_score.fill(num_treatment_samples / length)
        else:
            propensity_score = np.empty(length)
            propensity_score.fill(0.5)

        return propensity_score


class DataSetsHelper:
    """
    Wrapper class that saves a training, test and validation set for given training, test and validation DataFrames.
    """

    def __init__(self, df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame):
        """
        Creates an object containing the training, validation, and test set for a given dataset. This class contains utility methods for getting the full test, training and
        validation sets (including the treatment and response indicator), covariates and target sets.

        :param df_train: Dataframe containing the training set (including the treatment and response indicator)
        :param df_train: Dataframe containing the validation set (including the treatment and response indicator)
        :param df_test: Dataframe containing the test set (including the treatment and response indicator)
        """
        # Training
        self.df_train = df_train                                                        # Full DataFrame
        self.x_train = df_train.drop(['response', 'treatment'], axis=1).to_numpy()      # Covariates
        self.y_train = df_train['response'].to_numpy()                                  # Target

        # Test
        self.df_test = df_test                                                          # Full DataFrame
        self.x_test = df_test.drop(['response', 'treatment'], axis=1).to_numpy()        # Covariates
        self.y_test = df_test['response'].to_numpy()                                    # Target

        # Validation
        self.valid = True
        if df_valid.empty:
            self.valid = False
        else:
            self.df_valid = df_valid                                                    # Full DataFrame
            self.x_valid = df_valid.drop(['response', 'treatment'], axis=1).to_numpy()  # Covariates
            self.y_valid = df_valid['response'].to_numpy()                              # Target

    def get_covariates(self):
        return self.x_train, self.x_valid, self.x_test

    def get_targets(self):
        return self.y_train, self.y_valid, self.y_test

    def get_dataframes(self):
        return self.df_train, self.df_valid, self.df_test


class ApproachParameters:
    """
    Utility class that encompassees all parameters needed to create an approach instance.
    """

    def __init__(self, cost_sensitive: bool, feature_importance: bool, path: str, save: bool, split_number: int):
        """
        Utility class that encompassees all parameters needed to create an approach instance.

        :param cost_sensitive: Set this to true for cost sensitive learning.
        :param feature_importance: Set this to True to return the feature importances of the classifiers
        :param path: Path where the models generated during approach execution should be saved.
        :param save: Set this to True if during training the generated models should be saved
        :param split_number: Number of current split. Used for logging and saving purposes in order to identify saved models.

        """
        self.cost_sensitive = cost_sensitive
        self.feature_importance = feature_importance
        self.path = path
        self.save = save
        self.split_number = split_number
