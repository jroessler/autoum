import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

root = os.getenv("ROOT_FOLDER")
sys.path.append(root + "src/")

from approaches.bayesian_causal_forest import BayesianCausalForest
from approaches.generalized_random_forest import GeneralizedRandomForest
from approaches.lais_generalization import LaisGeneralization
from approaches.r_learner import RLearner
from approaches.class_variable_transformation import ClassVariableTransformation
from approaches.s_learner import SLearner
from approaches.traditional import Traditional
from approaches.x_learner import XLearner
from approaches.treatment_dummy import TreatmentDummy
from approaches.two_model import TwoModel
from approaches.uplift_random_forest import UpliftRandomForest
from approaches.helper.helper_approaches import DataSetsHelper, ApproachParameters
from const.const import *

import logging

logging = logging.getLogger(__name__)


class HelperPipeline:
    """
    Helper class with utility functions which support the Pipelines.
    """

    def __init__(self):
        # Define paths
        self.data_folder = root + "data/"

        # Hillstrom Email
        self.hillstrom_path_folder = self.data_folder + "hillstrom-email/"
        self.hillstrom_visit_path = self.hillstrom_path_folder + "Hillstrom_Email_visit.csv"
        self.hillstrom_w_visit_path = self.hillstrom_path_folder + "Hillstrom_Email_w_visit.csv"
        self.hillstrom_m_visit_path = self.hillstrom_path_folder + "Hillstrom_Email_m_visit.csv"
        self.hillstrom_conversion_path = self.hillstrom_path_folder + "Hillstrom_Email_conversion.csv"
        self.hillstrom_w_conversion_path = self.hillstrom_path_folder + "Hillstrom_Email_w_conversion.csv"
        self.hillstrom_m_conversion_path = self.hillstrom_path_folder + "Hillstrom_Email_m_conversion.csv"

        # Criteo
        self.criteo_path_folder = self.data_folder + "criteo-marketing/"
        self.criteo_path = self.criteo_path_folder + "criteo_uplift.csv"
        self.criteo_path_resampled = self.criteo_path_folder + "criteo_uplift_resampled.csv"

        # Starbucks
        self.starbucks_folder = self.data_folder + "starbucks/"
        self.starbucks_path = self.starbucks_folder + "starbucks.csv"

        # Company E
        self.companye_path_folder = self.data_folder + "company-e/confidential/"
        self.companye_b_path = self.companye_path_folder + "Companye_b.csv"
        self.companye_k_path = self.companye_path_folder + "Companye_k.csv"
        self.companye_path = self.companye_path_folder + "Companye.csv"

        # Bank telemarketing
        self.bank_telemarketing_path = self.data_folder + "bank-telemarketing/"
        self.bank_additional_path_1 = self.bank_telemarketing_path + "Bank-Telemarketing_1.csv"
        self.bank_additional_path_2 = self.bank_telemarketing_path + "Bank-Telemarketing_2.csv"

        # Social Pressure
        self.social_pressure_path = self.data_folder + "social-pressure/"
        self.social_pressure_neighbors_path = self.social_pressure_path + "social_pressure_neighbors.csv"

        # Open-data (based on MI campaign 1)
        self.open_data_folder = self.data_folder + "open-data/confidential/"
        self.open_data_original_path = self.open_data_folder + "open_data_original.csv"
        self.open_data_enriched_path = self.open_data_folder + "open_data_enriched.csv"

        # Lenta
        self.lenta_path_folder = self.data_folder + "lenta/"
        self.lenta_path = self.lenta_path_folder + "lenta-dataset.csv"

        # Criteo_v2
        self.criteo_v2_path_folder = self.data_folder + "criteo-marketing-v2/"
        self.criteo_v2_path = self.criteo_v2_path_folder + "criteo_uplift_v2.csv"
        self.criteo_v2_path_resampled = self.criteo_v2_path_folder + "criteo_uplift_v2_resampled.csv"

    def get_dataframe(self, dataset_name: str, test_size, random_seed):
        """
        Return the dataframe for the given name.

        :param dataset_name: Name of the dataset
        :param test_size: Size of the Test Set
        :param random_seed: Random seed
        :return: Dataframe
        """

        if dataset_name == "Hillstrom":
            data = pd.read_csv(self.hillstrom_visit_path)
        elif dataset_name == "Hillstrom_Conversion":
            data = pd.read_csv(self.hillstrom_conversion_path)
        elif dataset_name == "Hillstrom_Women":
            data = pd.read_csv(self.hillstrom_w_visit_path)
        elif dataset_name == "Hillstrom_Men":
            data = pd.read_csv(self.hillstrom_m_visit_path)
        elif dataset_name == "Hillstrom_Women_Conversion":
            data = pd.read_csv(self.hillstrom_w_conversion_path)
        elif dataset_name == "Hillstrom_Men_Conversion":
            data = pd.read_csv(self.hillstrom_m_conversion_path)
        elif dataset_name == "Criteo":
            data = pd.read_csv(self.criteo_path)
        elif dataset_name == "Criteo_Resampled":
            data = pd.read_csv(self.criteo_path_resampled)
        elif dataset_name == "Starbucks":
            data = pd.read_csv(self.starbucks_path)
        elif dataset_name == "Companye_b":
            data = pd.read_csv(self.companye_b_path)
        elif dataset_name == "Companye_k":
            data = pd.read_csv(self.companye_k_path)
        elif dataset_name == "Companye":
            data = pd.read_csv(self.companye_path)
        elif dataset_name == "Open_Data_Original":
            data = pd.read_csv(self.open_data_original_path)
        elif dataset_name == "Open_Data_Enriched":
            data = pd.read_csv(self.open_data_enriched_path)
        elif dataset_name == "Bank_This_Campaign":
            data = pd.read_csv(self.bank_additional_path_1)
        elif dataset_name == "Bank_Both_Campaigns":
            data = pd.read_csv(self.bank_additional_path_2)
        elif dataset_name == "Social_Pressure_Neighbors":
            data = pd.read_csv(self.social_pressure_neighbors_path)
        elif dataset_name == "Lenta":
            data = pd.read_csv(self.lenta_path)
        elif dataset_name == "Criteo_v2":
            data = pd.read_csv(self.criteo_v2_path)
        elif dataset_name == "Criteo_v2_Resampled":
            data = pd.read_csv(self.criteo_v2_path_resampled)
        else:
            logging.error("Please choose a proper dataset")
            return None, None

        try:
            df_train, df_test = train_test_split(data, test_size=test_size, shuffle=True, stratify=data[['response', 'treatment']], random_state=random_seed)
            df_train.reset_index(inplace=True, drop=True)
            df_test.reset_index(inplace=True, drop=True)
        except ValueError:
            logging.error("Stratification not possible" + data.groupby(["response", "treatment"]).size().reset_index(name="Counter").to_string())
            raise ValueError("Stratification not possible" + data.groupby(["response", "treatment"]).size().reset_index(name="Counter").to_string())

        return df_train, df_test

    @staticmethod
    def apply_approach(classifier, data_set_helper, feature_importance):
        """
        You can pass classifiers for every approach:
         - S-Learner
         - Traditional
         - Uplift Random Forest (with 8 different splitting criteria)
         - Class Variable Transformation
         - Lais Generalization
         - Two-Model
         - X-Learner
         - R-Learner
         - Treatment Dummy
         - Bayesian Causal Forest
         - Generalized Random Forest

        For parameter definition see :func:`~HelperPipeline.apply_uplift_approaches`
        """
        feature_importances_dict = {}
        logging.info(str(classifier.__class__.__name__))

        try:
            # Predict training, validation and test scores
            result_dict = classifier.analyze(data_set_helper)

            score_uplift_train = result_dict["score_train"]
            score_uplift_valid = result_dict["score_valid"]
            score_uplift_test = result_dict["score_test"]

            if feature_importance:
                feature_importances_dict = result_dict["feature_importance"]

        except Exception as e:
            logging.error(f"Error in {str(classifier.__class__.__name__)} Approach; the approach is no longer considered for further evaluation; error message: \n {e}")
            score_uplift_train, score_uplift_valid, score_uplift_test = np.nan, np.nan, np.nan

        return score_uplift_train, score_uplift_valid, score_uplift_test, feature_importances_dict

    @staticmethod
    def apply_uplift_approaches(df_train: pd.DataFrame,
                                df_valid: pd.DataFrame,
                                df_test: pd.DataFrame,
                                parameters: dict,
                                approach: list,
                                split_number: int,
                                cost_sensitive: bool = False,
                                feature_importance: bool = False,
                                save_models: bool = False) -> dict:
        """
        Apply given uplift modeling approaches on the given dataframes and return the scores

        :param df_train: Dataframe containing the training set including all covariates, response, and treatment variable.
        :param df_valid: Dataframe containing the validation set including all covariates, response, and treatment variable. Note, this DataFrame can also be empty for final
            model creation.
        :param df_test: Dataframe containing the test set including all covariates, response, and treatment variable.
        :param parameters: Parameter dictionary which contains the dictionaries of the various uplift modeling approaches
        :param approach: List containing the names of the approaches which should be evaluated.
        :param split_number: Number of current split. Used for logging and saving purposes in order to identify saved models.
        :param cost_sensitive: Set this to true for cost sensitive learning.
        :param feature_importance: Set this to True to return the feature importances of the classifiers
        :param save_models: True if the models generated during training shall be saved. False otherwise.
        :return: Dictionary with the following keys: df_scores_train, df_scores_test, df_train, df_test, feature_importances (empty dictionary if not used)
        """

        # Create DataFrames which contain results (i.e., the uplift scores)
        df_scores_train = pd.DataFrame()
        df_scores_valid = pd.DataFrame()
        df_scores_test = pd.DataFrame()

        # Dictionary which will be used for storing the feature importance of each classifier
        fid = {}

        # DataSetsHelper contains all base dataframes and their transformations. These are used in the analyze methods of the approaches.
        ds_helper = DataSetsHelper(df_train=df_train, df_valid=df_valid, df_test=df_test)
        # ApproachParameters contains alll parameters necessary to initialize an approach classifier
        approach_params = ApproachParameters(cost_sensitive=cost_sensitive, feature_importance=feature_importance, path=root, save=save_models, split_number=split_number)

        # This dictionary is used as wrapper for passing all parameters at once for apply_approach
        apply_params = {
            "data_set_helper": ds_helper,
            "feature_importance": feature_importance,
        }

        ### Apply Uplift Modeling Approaches ###
        start = time.time()

        # 1. UPLIFT RANDOM FOREST APPROACH
        # 1.1 WITH EUCLIDEAN DISTANCE
        if "URF_ED" in approach:
            urf_ed = UpliftRandomForest(parameters[URF_TITLE + "_parameters"], approach_params, eval_function="ED")
            df_scores_train[URF_ED_TITLE], df_scores_valid[URF_ED_TITLE], df_scores_test[URF_ED_TITLE], _ = HelperPipeline.apply_approach(urf_ed, **apply_params)

        # 1.2 WITH KULLBACK-LEIBLER DIVERGENCE
        if "URF_KL" in approach:
            urf_kl = UpliftRandomForest(parameters[URF_TITLE + "_parameters"], approach_params, eval_function="KL")
            df_scores_train[URF_KL_TITLE], df_scores_valid[URF_KL_TITLE], df_scores_test[URF_KL_TITLE], _ = HelperPipeline.apply_approach(urf_kl, **apply_params)

        # 1.3 WITH CHI-SQAURED DIVERGENCE
        if "URF_CHI" in approach:
            urf_chi = UpliftRandomForest(parameters[URF_TITLE + "_parameters"], approach_params, eval_function="Chi")
            df_scores_train[URF_CHI_TITLE], df_scores_valid[URF_CHI_TITLE], df_scores_test[URF_CHI_TITLE], _ = HelperPipeline.apply_approach(urf_chi, **apply_params)

        # 1.4 WITH DELTA-DELTA-PI Criterion
        if "URF_DDP" in approach:
            urf_ddp = UpliftRandomForest(parameters[URF_TITLE + "_parameters"], approach_params, eval_function="DDP")
            df_scores_train[URF_DDP_TITLE], df_scores_valid[URF_DDP_TITLE], df_scores_test[URF_DDP_TITLE], _ = HelperPipeline.apply_approach(urf_ddp, **apply_params)

        # 1.5 WITH CONTEXTUAL TREATMENT SELECTION
        if "URF_CTS" in approach:
            urf_cts = UpliftRandomForest(parameters[URF_TITLE + "_parameters"], approach_params, eval_function="CTS")
            df_scores_train[URF_CTS_TITLE], df_scores_valid[URF_CTS_TITLE], df_scores_test[URF_CTS_TITLE], _ = HelperPipeline.apply_approach(urf_cts, **apply_params)

        # 1.6 IT CRITERION
        if "URF_IT" in approach:
            urf_it = UpliftRandomForest(parameters[URF_TITLE + "_parameters"], approach_params, eval_function="IT")
            df_scores_train[URF_IT_TITLE], df_scores_valid[URF_IT_TITLE], df_scores_test[URF_IT_TITLE], _ = HelperPipeline.apply_approach(urf_it, **apply_params)

        # 1.7 CIT CRITERION
        if "URF_CIT" in approach:
            urf_cit = UpliftRandomForest(parameters[URF_TITLE + "_parameters"], approach_params, eval_function="CIT")
            df_scores_train[URF_CIT_TITLE], df_scores_valid[URF_CIT_TITLE], df_scores_test[URF_CIT_TITLE], _ = HelperPipeline.apply_approach(urf_cit, **apply_params)

        # 2. S-LEARNER
        if "SLEARNER" in approach:
            s_learner = SLearner(parameters[SLEARNER_TITLE + "_parameters"], approach_params)
            df_scores_train[SLEARNER_TITLE], df_scores_valid[SLEARNER_TITLE], df_scores_test[SLEARNER_TITLE], fid[SLEARNER_TITLE] = HelperPipeline.apply_approach(s_learner, **apply_params)

        # 3. CLASS VARIABLE TRANSFORMATION
        if "CVT" in approach:
            cvt = ClassVariableTransformation(parameters[CVT_TITLE + "_parameters"], approach_params)
            df_scores_train[CVT_TITLE], df_scores_valid[CVT_TITLE], df_scores_test[CVT_TITLE], fid[CVT_TITLE] = HelperPipeline.apply_approach(cvt, **apply_params)

        # 4. LAIS OUTCOME APPROACH
        if "LAIS" in approach:
            lais = LaisGeneralization(parameters[LAIS_TITLE + "_parameters"], approach_params)
            df_scores_train[LAIS_TITLE], df_scores_valid[LAIS_TITLE], df_scores_test[LAIS_TITLE], fid[LAIS_TITLE] = HelperPipeline.apply_approach(lais, **apply_params)

        # 5. TWO-MODEL APPROACH
        if "TWO_MODEL" in approach:
            two_model = TwoModel(parameters[TWO_MODEL_TITLE + "_parameters"], approach_params)
            df_scores_train[TWO_MODEL_TITLE], df_scores_valid[TWO_MODEL_TITLE], df_scores_test[TWO_MODEL_TITLE], fid_tm = HelperPipeline.apply_approach(two_model, **apply_params)

            fid.update(fid_tm)

        # 6. TRADITIONAL APPROACH
        if "TRADITIONAL" in approach:
            traditional = Traditional(parameters[TRADITIONAL_TITLE + "_parameters"], approach_params)
            df_scores_train[TRADITIONAL_TITLE], df_scores_valid[TRADITIONAL_TITLE], df_scores_test[TRADITIONAL_TITLE], fid[TRADITIONAL_TITLE] = HelperPipeline.apply_approach(
                traditional, **apply_params)

        # 7. X-LEARNER
        if "XLEARNER" in approach:
            x_learner = XLearner(parameters[XLEARNER_TITLE + "_parameters"], approach_params)
            df_scores_train[XLEARNER_TITLE], df_scores_valid[XLEARNER_TITLE], df_scores_test[XLEARNER_TITLE], _ = HelperPipeline.apply_approach(x_learner, **apply_params)

        # 8. R-LEARNER
        if "RLEARNER" in approach:
            r_learner = RLearner(parameters[RLEARNER_TITLE + "_parameters"], approach_params)
            df_scores_train[RLEARNER_TITLE], df_scores_valid[RLEARNER_TITLE], df_scores_test[RLEARNER_TITLE], _ = HelperPipeline.apply_approach(r_learner, **apply_params)

        # 9. TREATMENT DUMMY
        if "TREATMENT_DUMMY" in approach:
            tda = TreatmentDummy(parameters[TREATMENT_DUMMY_TITLE + "_parameters"], approach_params)
            df_scores_train[TREATMENT_DUMMY_TITLE], df_scores_valid[TREATMENT_DUMMY_TITLE], df_scores_test[TREATMENT_DUMMY_TITLE], _ = HelperPipeline.apply_approach(tda, **apply_params)

        # 10. GENERALIZED RANDOM FOREST
        if "GRF" in approach:
            grf = GeneralizedRandomForest(parameters[GRF_TITLE + "_parameters"], approach_params)
            df_scores_train[GRF_TITLE], df_scores_valid[GRF_TITLE], df_scores_test[GRF_TITLE], _ = HelperPipeline.apply_approach(grf, **apply_params)

        # 11. BAYESIAN CAUSAL FOREST
        if "BCF" in approach:
            bcf = BayesianCausalForest(parameters[BCF_TITLE + "_parameters"], approach_params)
            df_scores_train[BCF_TITLE], df_scores_valid[BCF_TITLE], df_scores_test[BCF_TITLE], _ = HelperPipeline.apply_approach(bcf, **apply_params)

        appr_name = approach[0]

        end = time.time()
        logging.info(f'Function {appr_name} took {(end - start):.2f} s')

        df_scores_train['response'] = df_train['response'].to_numpy()
        df_scores_train['treatment'] = df_train['treatment'].to_numpy()

        df_scores_valid['response'] = df_valid['response'].to_numpy()
        df_scores_valid['treatment'] = df_valid['treatment'].to_numpy()

        df_scores_test['response'] = df_test['response'].to_numpy()
        df_scores_test['treatment'] = df_test['treatment'].to_numpy()

        return {
            "df_scores_train": df_scores_train,
            "df_scores_valid": df_scores_valid,
            "df_scores_test": df_scores_test,
            "feature_importances": fid
        }

    @staticmethod
    def cast_to_dataframe(_ds):
        """
        Cast the given dictionaries to one dataframe

        :param _ds: List of dictionaries
        :return: DataFrame
        """

        if isinstance(_ds, list):
            final_dict = defaultdict(list)

            # Iterate through each dictionary in ds
            for d in _ds:
                for key, value in d.items():
                    final_dict[key].append(value)
            # Cast back to dict
            final_dict = dict(final_dict)

            df = pd.DataFrame.from_dict(final_dict)
        else:
            df = pd.DataFrame(_ds, index=[0])
        return df

    @staticmethod
    def save_feature_importance(importances, feature_names, title, save_figure, plot_figure):
        """
        Save feature importance plot

        :param importances: Importances
        :param feature_names: Names of all features
        :param title: Title of the plot
        :param save_figure: True, if figure should be saved. False otherwise
        :param plot_figure: True, if figure should be plotted. False otherwise
        """
        indices = np.argsort(importances)

        if plot_figure:
            plt.figure(figsize=(14, 8))
            plt.title(title)
            plt.barh(range(len(indices)), importances[indices], color='b', align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Relative Importance')

        now = datetime.now()
        now_date = now.date()
        if save_figure:
            plt.savefig(os.getenv("ROOT_FOLDER") + "results/figures/" + title + "_" + str(now_date) + ".png")

        if plot_figure:
            plt.show()
