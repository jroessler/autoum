import logging
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()
root = os.getenv("ROOT_FOLDER")
sys.path.append(root + "src/")

from evaluation.evaluation import UpliftEvaluation
from const.const import *
from pipelines.helper.helper_pipeline import HelperPipeline

pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 10)


class PipelineRW:
    """
    Pipeline with real-world datasets using multiple customer targeting approaches
    """

    def __init__(self,
                 bins=10,
                 cost_sensitive=False,
                 cv_number_splits=5,
                 feature_importance=False,
                 fontsize=14,
                 honesty=False,
                 logging_file=False,
                 logging_level=1,
                 max_depth=25,
                 max_features='auto',
                 metrics_calculate_absolute=False,
                 metrics_qini_coefficient=False,
                 metrics_save_metrics=False,
                 min_samples_leaf=50,
                 min_samples_treatment=10,
                 normalization=True,
                 n_estimators=200,
                 n_jobs_=20,
                 n_reg=100,
                 plot_figures=True,
                 plot_optimum=False,
                 plot_grayscale=False,
                 plot_uqc=True,
                 plot_save_figures=False,
                 pool_capacity=40,
                 run_name="RUN",
                 run_id=1,
                 random_seed=123,
                 save_models=False,
                 show_title=False,
                 test_size=0.2,
                 validation_size=0.2):
        """
        Creates a pipeline that can be used to analyze real-world data sets
        fontsize, show_title
        :param bins: Number of bins. Default: 10
        :param cost_sensitive: True if cost sensitive learning shall be performed during training. False otherwise. Default: False
        :param cv_number_splits: If cv_number_splits == 2, use a single training/test split with validation_size. Otherwise, use cv_number_splits folds. Default: 5
        :param feature_importance: True if the feature importances of the classifiers shall be returned. False otherwise. Default: False
        :param fontsize: Size of each element in the graphics. Default: 14.
        :param honesty: True if the honest approach based on "Athey, S., & Imbens, G. (2016) shall be performed - Note: This only applies to direct uplift approaches.
            Default: False
        :param logging_file: True if logs shall be saved to file. False otherwise. Default: False
        :param logging_level: Pass 1 for only logging warning, 2 for also logging informational statements and 3 for also logging debugging statements. Default: 1
        :param max_depth: The maximum depth of a decision tree that is created during training. Default: 25
        :param max_features: The number os features that shall be chosen randomly from all features when looking for the best split during decision tree creation. Default: 'auto'
        :param metrics_calculate_absolute: True if metrics should be calculated in absolute numbers. False, if they should be calculated in or relative numbers. Default: False
        :param metrics_qini_coefficient: True if the qini coefficient metric (Radcliffe) should be calculated (additionally). Default: False
        :param metrics_save_metrics: True if the qini-related metrics shall be saved to file. Default: True
        :param min_samples_leaf: The minimum number of data records required to be split at a leaf node. Default: 50
        :param min_samples_treatment: The minimum number of data records from the treatment group required to be split at a leaf node. Default: 10
        :param normalization: True, if normalization should be applied to some of the Direct approaches (ED, KL, CHI). False otherwise. Default: True.
        :param n_estimators: The number of estimators (trees) that shall be created in the random forest. Default: 200
        :param n_jobs_: The number of models that can be calculated dependent on the number of free kernels. This parameter will only be used if the cv_number_splits is 2.
            Note: For model creation the data set which is used for training is copied n_jobs_ times. Thus, this parameter is heavily limited by size of available RAM.
            Set this parameter with care. Default: 20
        :param n_reg: (Only for Direct uplift approaches) This represents the regularization parameter defined in Rzepakowski et al. 2012.
            It is the weight (in terms of sample size) of the parent node influence on the child node. Default: 100
        :param plot_figures: True if the qini curves shall be plotted. False otherwise. Default: True
        :param plot_optimum: True if the optimium qini curve shall be plotted. False otherwise. Default: False
        :param plot_grayscale: True if the curves shall be plotted in grayscale. False otherwise. Default: False
        :param plot_uqc: True if the UQC value for a curve should be included in the plot legend. False otherwise. Default: True
        :param plot_save_figures: True if the resulting qini figures shall be saved. False otherwise. Default: False
        :param pool_capacity: Set this to the maximum number of free kernels for the calculation. Default 40
        :param run_id: Id of the run (For logging and saving purposes). Default: 1
        :param run_name: Name of the run (For logging and saving purposes). Default: "RUN"
        :param random_seed: The integer random_seed will ensure that the splits created will be the same in every run. If the k splits shall be created randomly, set random_seed
            to None. Default: 123
        :param save_models: True if the models generated during training shall be saved. False otherwise. Default: False
        :param show_title: True, if the graphics should contain a title. False otherwise. Default: False.
        :param test_size: Size of the Test set. Default: 0.2
        :param validation_size: Only use this parameter if cv_number_splits == 2. Sets the size of the validation set for a single split. Default: 0.2
        """

        self.parameters = None
        self.run_name = run_name + "_" + str(run_id)
        self.bins = bins
        self.cost_sensitive = cost_sensitive
        self.cv_number_splits = cv_number_splits
        self.metrics_calculate_absolute = metrics_calculate_absolute
        self.dataset = ""
        self.feature_importance = feature_importance
        self.fontsize = fontsize
        self.metrics_qini_coefficient = metrics_qini_coefficient
        self.metrics_save_metrics = metrics_save_metrics
        self.plot_figures = plot_figures
        self.plot_optimum = plot_optimum
        self.plot_grayscale = plot_grayscale
        self.plot_uqc = plot_uqc
        self.plot_save_figures = plot_save_figures
        self.pool_capacity = pool_capacity
        self.random_seed = random_seed
        self.save_models = save_models
        self.show_title = show_title
        self.test_size = test_size
        self.validation_size = validation_size
        self.n_estimators = n_estimators

        # Number of jobs running in parallel (for calculating the trees). Increasing the number of jobs might reduce the time for training and testing the models
        # Only use this parameter if self.cv_number_splits == 2
        if self.cv_number_splits == 2:
            n_jobs = n_jobs_
        else:
            n_jobs = 1

        # Hyperparameters of different uplift modeling approaches
        self.max_features = max_features
        self.set_parameters(n_estimators, max_depth, min_samples_leaf, min_samples_treatment, n_reg, n_jobs, normalization, honesty, random_seed)

        # Create helper
        self.helper = HelperPipeline()

        # Create logger
        log_level = logging.WARNING
        if logging_level == 2:
            log_level = logging.INFO
        elif logging_level == 3:
            log_level = logging.DEBUG

        if logging_file:
            filename = root + 'logging/' + self.run_name + '.log'
            logging.basicConfig(filename=filename, level=log_level)
        else:
            logging.basicConfig(level=log_level)

        logging.info(
            f"run_name: {run_name}, run_id: {run_id}, max_depth: {max_depth}, n_estimators: {n_estimators}, min_samples_leaf: {min_samples_leaf}, min_samples_treatment: {min_samples_treatment}")

        self.sanity_checks()

    def sanity_checks(self):
        """
        Check arguments when creating the PipelineRW instance and raise error if the users have entered invalid entries
        """
        assert 5 <= self.bins <= 100, "Please select 5 <= bins <= 100"
        assert 2 <= self.cv_number_splits <= 10, "Please select 2 <= cv_number_splits <= 10"
        assert 10 <= self.fontsize <= 20, "Please select 10 <= fontsize <= 20"
        assert 0.1 <= self.test_size <= 0.9, "Please select 0.1 <= test_size <= 0.9"
        assert 0.1 <= self.validation_size <= 0.9, "Please select 0.1 <= validation_size <= 0.9"
        assert self.n_estimators % 4 == 0, "Please select a multiple of 4 as n_estimators"

    def analyze_dataset(self, dataset_name):
        """
        Apply, compare, and evaluate various uplift modeling approaches on the given data set.

        Datasets which should be analyzed. You can choose among the following options:
         - Hillstrom, Hillstrom_Women, Hillstrom_Men, Hillstrom_Conversion, Hillstrom_Women_Conversion, Hillstrom_Men_Conversion,
         - Criteo, Criteo_Resampled, Criteo_v2, Criteo_v2_Resampled
         - Starbucks,
         - Companye_b, Companye_k, Companye,
         - (Quasi experimental data; not an A/B trial) Bank_This_Campaign, Bank_Both_Campaigns
         - Social_Pressure_Neighbors
         - Lenta

        :param dataset_name: Name of the data set
        """

        # Set dataset name
        self.dataset = dataset_name

        start = time.time()
        logging.info("Dataset: {}".format(dataset_name))

        # Get training and test Set
        df_train, df_test = self.helper.get_dataframe(dataset_name, test_size=self.test_size, random_seed=self.random_seed)

        # Data set not found
        if df_train is None:
            return

        # Get feature names
        feature_names = list(df_train.drop(['response', 'treatment'], axis=1).columns.values)

        # If self.max_features is set to auto, set the number of features for the direct uplift approach to sqrt(n_features)
        if self.max_features == 'auto':
            n_features = len(feature_names)
            self.parameters[URF_TITLE + '_parameters']['max_features'] = int(round(np.sqrt(n_features)))

        # k-fold cross validation
        if self.cv_number_splits > 2:
            dict_uplift = self.analyze_k_fold_cv(df_train=df_train, df_test=df_test)

        # regular train/test split
        else:
            dict_uplift = self.analyze_single_fold(df_train=df_train, df_test=df_test)

        # Calculate, plot and save metrics
        self.calculate_metrics(list_feature_importances=dict_uplift["feature_importances"],
                               feature_names=feature_names,
                               dataset_name=dataset_name,
                               list_dict_uplift_train=dict_uplift["list_dict_uplift_train"],
                               list_dict_uplift_valid=dict_uplift["list_dict_uplift_valid"],
                               list_dict_uplift_test=dict_uplift["list_dict_uplift_test"],
                               list_dict_opt_uplift_train=dict_uplift["list_dict_opt_uplift_train"],
                               list_dict_opt_uplift_valid=dict_uplift["list_dict_opt_uplift_valid"],
                               list_dict_opt_uplift_test=dict_uplift["list_dict_opt_uplift_test"])

        end = time.time()
        logging.info('Function took %0.3f s' % (end - start))

    def analyze_k_fold_cv(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        """
        Train various approaches using k-fold cv on the given Training DataFrame.
        Final evaluation is based on the given Test DataFrame.

        :param df_train: Trainings DataFrame
        :param df_test: Test DataFrame
        :return: dictionary containg the lists of dictionaries which contain the uplift values for each bin/decile for training, validation, and test
        """
        dataframe_pairs = self.create_k_splits(df_train=df_train, df_test=df_test)
        approach_tuples = PipelineRW.create_approach_tuples(dataframe_pairs)

        # Use multiprocessing to analyze multiple splits and approaches in parallel
        pool = Pool(processes=self.pool_capacity)
        ds = pool.map(self.train_eval_splits, approach_tuples)
        pool.close()

        # Pool stores multiple return values in just one value (like an array)
        list_dict_uplift_train = [result[0] for result in ds]
        list_dict_uplift_valid = [result[1] for result in ds]
        list_dict_uplift_test = [result[2] for result in ds]
        list_dict_opt_uplift_train = [result[3] for result in ds]
        list_dict_opt_uplift_valid = [result[4] for result in ds]
        list_dict_opt_uplift_test = [result[5] for result in ds]
        feature_importances = [result[6] for result in ds]

        return {
            "list_dict_uplift_train": list_dict_uplift_train,
            "list_dict_uplift_valid": list_dict_uplift_valid,
            "list_dict_uplift_test": list_dict_uplift_test,
            "list_dict_opt_uplift_train": list_dict_opt_uplift_train,
            "list_dict_opt_uplift_valid": list_dict_opt_uplift_valid,
            "list_dict_opt_uplift_test": list_dict_opt_uplift_test,
            "feature_importances": feature_importances
            }

    def analyze_single_fold(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        """
        Analyze the given DataFrame using a single train/test split

        :param df_train: Trainings DataFrame
        :param df_test: Test DataFrame
        :return: dictionary containg the lists of dictionaries which contain the uplift values for each bin/decile for training, validation, and test
        """
        # Create a single train/test split
        df_train, df_valid = self.create_single_split(df_train)
        param_list = [1, df_train, df_valid, df_test]

        # Create an approach tuple list that contains the name of the approach to be calculated
        approach_tuple_list = []
        for approach in PipelineRW.create_approach_list_for_single_split():
            approach_tuple_list.append([approach])

        list_dict_uplift_train = []
        list_dict_uplift_valid = []
        list_dict_uplift_test = []
        list_dict_opt_uplift_train = []
        list_dict_opt_uplift_valid = []
        list_dict_opt_uplift_test = []
        feature_importances = []

        # Analyze the training and test data set with every approach contained in the approach_tuple_list
        for approach_tuple in approach_tuple_list:
            list_tmp = param_list.copy()
            list_tmp.extend(approach_tuple)
            uplift_train, uplift_valid, uplift_test, opt_uplift_train, opt_uplift_valid, opt_uplift_test, feature_importance = self.train_eval_splits(tuple(list_tmp))
            # Add the results for one approach to the list that contains all results
            list_dict_uplift_train.append(uplift_train)
            list_dict_uplift_valid.append(uplift_valid)
            list_dict_uplift_test.append(uplift_test)
            list_dict_opt_uplift_train.append(opt_uplift_train)
            list_dict_opt_uplift_valid.append(opt_uplift_valid)
            list_dict_opt_uplift_test.append(opt_uplift_test)
            feature_importances.append(feature_importance)

        return {
            "list_dict_uplift_train": list_dict_uplift_train,
            "list_dict_uplift_valid": list_dict_uplift_valid,
            "list_dict_uplift_test": list_dict_uplift_test,
            "list_dict_opt_uplift_train": list_dict_opt_uplift_train,
            "list_dict_opt_uplift_valid": list_dict_opt_uplift_valid,
            "list_dict_opt_uplift_test": list_dict_opt_uplift_test,
            "feature_importances": feature_importances
            }

    def create_k_splits(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        """
        Create k training/validation splits for a given dataset which can be used for k-fold cross validation

        :param df_train: DataFrame (entire Trainings data set)
        :param df_test: Test DataFrame
        :return: List of DataFrame pairs
        """
        # Stratified k Fold CV
        kfold = StratifiedKFold(n_splits=self.cv_number_splits, shuffle=True, random_state=self.random_seed)

        # In order to enable a stratified split, create a new column group representing any arbitrary combination of treatment and response
        df_train.loc[((df_train['treatment'] == 0) & (df_train['response'] == 0)), 'group'] = 0
        df_train.loc[((df_train['treatment'] == 1) & (df_train['response'] == 0)), 'group'] = 1
        df_train.loc[((df_train['treatment'] == 0) & (df_train['response'] == 1)), 'group'] = 2
        df_train.loc[((df_train['treatment'] == 1) & (df_train['response'] == 1)), 'group'] = 3
        y = df_train["group"].to_numpy()

        # Create dataframe pairs for each split
        dataframe_pairs = []
        try:
            for i, (idx_train, idx_valid) in enumerate(kfold.split(df_train, y)):
                df_train_ = df_train.iloc[idx_train].copy()
                df_valid_ = df_train.iloc[idx_valid].copy()

                df_train_.drop(["group"], axis=1, inplace=True)
                df_valid_.drop(["group"], axis=1, inplace=True)

                df_train_.reset_index(drop=True, inplace=True)
                df_valid_.reset_index(drop=True, inplace=True)

                dataframe_pairs.append(tuple((i, df_train_, df_valid_, df_test)))
        except ValueError:
            logging.error("Stratification not possible" + df_train.groupby(["response", "treatment"]).size().reset_index(name="Counter").to_string())
            raise ValueError("Stratification not possible" + df_train.groupby(["response", "treatment"]).size().reset_index(name="Counter").to_string())

        return dataframe_pairs

    def create_single_split(self, df: pd.DataFrame):
        """
        Create a single training/test split for a given dataset

        :param df: DataFrame (entire data set)
        :return: Two dataFrames (training and test)
        """
        try:
            df_train, df_valid = train_test_split(df, test_size=self.validation_size, shuffle=True, stratify=df[['response', 'treatment']], random_state=self.random_seed)
        except ValueError:
            logging.error("Stratification not possible" + df.groupby(["response", "treatment"]).size().reset_index(name="Counter").to_string())
            raise ValueError("Stratification not possible" + df.groupby(["response", "treatment"]).size().reset_index(name="Counter").to_string())

        return df_train, df_valid

    def train_eval_splits(self, args):
        """
        Apply different uplift modeling approaches for the given train/test split and return qini related metrics (qini curve with uplift values for each decile)

        :param args: Tuple containing:
        - index 0: The current split number (of k-fold CV)
        - index 1: Trainings DataFrame
        - index 2: Validation DataFrame
        - index 3: Test DataFrame
        - index 4: Name of the approach which shall be analyzed. See const.py for the names of the approaches.

        :returns:
        - dict_uplift_train - Training dictionary containing the uplift values for each decile and approach
        - dict_uplift_train - Validation dictionary containing the uplift values for each decile and approach
        - dict_uplift_test - Test dictionary containing the uplift values for each decile and approach
        - dict_opt_uplift_train - Training dictionaries containing the optimal uplift values for each decile and approach
        - dict_opt_uplift_train - Validation dictionaries containing the optimal uplift values for each decile and approach
        - dict_opt_uplift_test - Test dictionaries containing the optimal uplift values for each decile and
        - dict_feature_importances - Dictionary containing the feature importances for each approach
        """

        i = args[0]  # The current split number (i)
        df_train = args[1]  # Trainings dataframe
        df_valid = args[2]  # Trainings dataframe
        df_test = args[3]  # Test dataframe
        approach_name = None

        # If the args parameter does not contain *approach_name*, the method is called for a single train/test split (no k-fold cv) and thus, all approaches can be trained
        # simultaneously
        if len(args) > 4:
            approach_name = args[4]

        logging.info("Start Training. Split number {}".format(i))

        scores_dict = HelperPipeline.apply_uplift_approaches(df_train=df_train, df_valid=df_valid, df_test=df_test, parameters=self.parameters, approach=[approach_name],
                                                             split_number=i, cost_sensitive=self.cost_sensitive, feature_importance=self.feature_importance,
                                                             save_models=self.save_models)

        logging.info("Start Evaluation. Split number {}".format(i))

        # Get scores
        df_scores_train = scores_dict["df_scores_train"]
        df_scores_valid = scores_dict["df_scores_valid"]
        df_scores_test = scores_dict["df_scores_test"]

        # Calculate uplift values in each decile
        dict_uplift_train = UpliftEvaluation.calculate_actual_uplift_in_bins(df_scores_train, absolute=self.metrics_calculate_absolute, bins=self.bins)
        dict_uplift_valid = UpliftEvaluation.calculate_actual_uplift_in_bins(df_scores_valid, absolute=self.metrics_calculate_absolute, bins=self.bins)
        dict_uplift_test = UpliftEvaluation.calculate_actual_uplift_in_bins(df_scores_test, absolute=self.metrics_calculate_absolute, bins=self.bins)

        # Calculate optimal uplift values in each decile
        dict_opt_uplift_train = {}
        dict_opt_uplift_valid = {}
        dict_opt_uplift_test = {}

        if self.metrics_qini_coefficient or self.plot_optimum:
            if (type(args[-1]) == bool and args[-1] is True) or type(args[-1]) != bool:
                dict_opt_uplift_train = UpliftEvaluation.calculate_optimal_uplift_in_bins(df_scores_train, absolute=self.metrics_calculate_absolute, bins=self.bins)
                dict_opt_uplift_valid = UpliftEvaluation.calculate_optimal_uplift_in_bins(df_scores_valid, absolute=self.metrics_calculate_absolute, bins=self.bins)
                dict_opt_uplift_test = UpliftEvaluation.calculate_optimal_uplift_in_bins(df_scores_test, absolute=self.metrics_calculate_absolute, bins=self.bins)

        return dict_uplift_train, dict_uplift_valid, dict_uplift_test, dict_opt_uplift_train, dict_opt_uplift_valid, dict_opt_uplift_test, scores_dict["feature_importances"]

    def calculate_metrics(self,
                          list_feature_importances,
                          list_dict_uplift_train,
                          list_dict_uplift_valid,
                          list_dict_uplift_test,
                          list_dict_opt_uplift_train,
                          list_dict_opt_uplift_valid,
                          list_dict_opt_uplift_test,
                          feature_names,
                          dataset_name,):
        """
        Calculate qini related metrics such as unscaled Qini Coefficient (UQC), Qini Coefficient (QC), and Qini Curves

        :param list_feature_importances: List of feature importances (of each approach)
        :param list_dict_uplift_train: List of dictionaries which containg the uplift scores for each bin/deciles on the Trainings set
        :param list_dict_uplift_valid: List of dictionaries which containg the uplift scores for each bin/deciles on the Validation set
        :param list_dict_uplift_test: List of dictionaries which containg the uplift scores for each bin/deciles on the Test set
        :param list_dict_opt_uplift_train: List of dictionaries which containg the optimal uplift scores for each bin/deciles on the Trainings set
        :param list_dict_opt_uplift_valid: List of dictionaries which containg the optimal uplift scores for each bin/deciles on the Validation set
        :param list_dict_opt_uplift_test: List of dictionaries which containg the optimal uplift scores for each bin/deciles on the Test set
        :param feature_names: Name of the features / covariates
        :param dataset_name: Name of the dataset
        """

        if len(list_dict_uplift_test) == 0 and len(list_dict_uplift_train) == 0:
            return

        # Calculate mean of feature importances
        if self.feature_importance:
            self.calculate_feature_importance_mean(list_feature_importances, feature_names, dataset_name)

        # Cast all list with dictionaries into one dataframe (one for training and one for testing)
        df_uplift_train = HelperPipeline.cast_to_dataframe(list_dict_uplift_train)
        df_uplift_valid = HelperPipeline.cast_to_dataframe(list_dict_uplift_valid)
        df_uplift_test = HelperPipeline.cast_to_dataframe(list_dict_uplift_test)
        df_optimal_uplift_train = HelperPipeline.cast_to_dataframe(list_dict_opt_uplift_train)
        df_optimal_uplift_valid = HelperPipeline.cast_to_dataframe(list_dict_opt_uplift_valid)
        df_optimal_uplift_test = HelperPipeline.cast_to_dataframe(list_dict_opt_uplift_test)

        # Calculate single number metrics
        # Unscaled Qini Coefficient (Radcliffe, 2011)
        df_metrics_train = UpliftEvaluation.calculate_unscaled_qini_coefficient(df_uplift_train, bins=self.bins)
        df_metrics_valid = UpliftEvaluation.calculate_unscaled_qini_coefficient(df_uplift_valid, bins=self.bins)
        df_metrics_test = UpliftEvaluation.calculate_unscaled_qini_coefficient(df_uplift_test, bins=self.bins)

        # Qini Coefficient (Radcliffe, 2007) with Optimum Qini Curve which considers negative effects
        if self.metrics_qini_coefficient:
            df_metrics_train = UpliftEvaluation.calculate_qini_coefficient(df_metrics_train, df_optimal_uplift_train, bins=self.bins, num_columns=self.bins + 2)
            df_metrics_valid = UpliftEvaluation.calculate_qini_coefficient(df_metrics_valid, df_optimal_uplift_valid, bins=self.bins, num_columns=self.bins + 2)
            df_metrics_test = UpliftEvaluation.calculate_qini_coefficient(df_metrics_test, df_optimal_uplift_test, bins=self.bins, num_columns=self.bins + 2)

        # Calculate mean values (for better comparison and plotting the mean qini curves)
        df_metrics_mean_train = UpliftEvaluation.calculate_mean(df_metrics_train, self.bins, self.metrics_qini_coefficient)
        df_metrics_mean_valid = UpliftEvaluation.calculate_mean(df_metrics_valid, self.bins, self.metrics_qini_coefficient)
        df_metrics_mean_test = UpliftEvaluation.calculate_mean(df_metrics_test, self.bins, self.metrics_qini_coefficient)

        directory = f"{root}results/metrics/{self.run_name}/"
        prefix = f"{directory}{self.dataset}_{self.cv_number_splits}splits_"
        if self.metrics_save_metrics:
            if not os.path.exists(directory):
                os.makedirs(directory)
            df_optimal_uplift_train.to_csv(f"{prefix}opt_uplift_train.csv", index=False)
            df_optimal_uplift_valid.to_csv(f"{prefix}opt_uplift_valid.csv", index=False)
            df_optimal_uplift_test.to_csv(f"{prefix}opt_uplift_test.csv", index=False)
            df_metrics_train.to_csv(f"{prefix}metrics_train.csv", index=False)
            df_metrics_valid.to_csv(f"{prefix}metrics_valid.csv", index=False)
            df_metrics_test.to_csv(f"{prefix}metrics_test.csv", index=False)
            df_metrics_mean_train.to_csv(f"{prefix}mean_metrics_train.csv", index=False)
            df_metrics_mean_valid.to_csv(f"{prefix}mean_metrics_valid.csv", index=False)
            df_metrics_mean_test.to_csv(f"{prefix}mean_metrics_test.csv", index=False)

        # Plot Qini Curve
        if self.plot_figures:

            # Training
            self.plotting(df_metrics_mean_train, df_optimal_uplift_train)
            # Validation
            self.plotting(df_metrics_mean_valid, df_optimal_uplift_valid, split="Valid")
            # Test
            self.plotting(df_metrics_mean_test, df_optimal_uplift_test, split="Test")

    def plotting(self, df_metrics, df_optimal_uplift, split="Train"):
        """
        Plot (average) qini curve for each approach

        :param df_optimal_uplift: DataFrame containing the optimal uplift values for each bin/decile, split,  and approach
        :param df_metrics: DataFrame containing qini related metrics (e.g., unscaled qini coefficient) for each split and approach
        :param split: Name of the current split. Either "Training", "Valid", or "Test".
        """
        logging.info("Start Plotting")

        UpliftEvaluation.plot_qini_curve(df_metrics=df_metrics, df_opt_uplift_bins=df_optimal_uplift, plot_optimum=self.plot_optimum,
                                         fontsize=self.fontsize, absolute=self.metrics_calculate_absolute, title=f"{self.run_name} {self.dataset} Qini Curves {split}",
                                         show_title=self.show_title, save_figure=self.plot_save_figures, path=root, plot_grayscale=self.plot_grayscale, plot_uqc=self.plot_uqc,
                                         bins=self.bins, qc=self.metrics_qini_coefficient)

    @staticmethod
    def create_tuples(t, approach, opt):
        """
        Create a tuple containing (1) trainings dataframe, (2) validation dataframe, (3) test dataframe, (4) approach name (e.g., 'TWO-MODEL'), (5) boolean for optimum

        :param t: Tuple containing, trainings, validation, and test dataframe
        :param approach: Name of the approach (string)
        :param opt: True, if the optimum should be calculated for this approach. False otherwise
        :return: Tuple
        """
        y = list(t)
        y.append(approach)

        if not opt:
            y.append(True)
            opt = True
        else:
            y.append(False)

        return tuple(y), opt

    @staticmethod
    def create_approach_tuples(dataframe_pairs: list):
        """
        Create tuples for each split and approach in order to execute them in parallel. A tuple looks like the following:
        (1) trainings dataframe
        (2) validation dataframe
        (3) test dataframe
        (4) approach name (e.g., 'TWO-MODEL')

        :param dataframe_pairs: List of tuples containing data sets for splits
        :return: List of approach and dataframes that shall be processed
        """

        tuple_list = []

        for t in dataframe_pairs:
            if SLEARNER:
                y = list(t)
                y.append("SLEARNER")
                tuple_list.append(tuple(y))
            if CVT:
                y = list(t)
                y.append("CVT")
                tuple_list.append(tuple(y))
            if TWO_MODEL:
                y = list(t)
                y.append("TWO_MODEL")
                tuple_list.append(tuple(y))
            if LAIS:
                y = list(t)
                y.append("LAIS")
                tuple_list.append(tuple(y))
            if URF_ED:
                y = list(t)
                y.append("URF_ED")
                tuple_list.append(tuple(y))
            if URF_KL:
                y = list(t)
                y.append("URF_KL")
                tuple_list.append(tuple(y))
            if URF_CHI:
                y = list(t)
                y.append("URF_CHI")
                tuple_list.append(tuple(y))
            if URF_DDP:
                y = list(t)
                y.append("URF_DDP")
                tuple_list.append(tuple(y))
            if URF_CTS:
                y = list(t)
                y.append("URF_CTS")
                tuple_list.append(tuple(y))
            if URF_IT:
                y = list(t)
                y.append("URF_IT")
                tuple_list.append(tuple(y))
            if URF_CIT:
                y = list(t)
                y.append("URF_CIT")
                tuple_list.append(tuple(y))
            if XLEARNER:
                y = list(t)
                y.append("XLEARNER")
                tuple_list.append(tuple(y))
            if RLEARNER:
                y = list(t)
                y.append("RLEARNER")
                tuple_list.append(tuple(y))
            if TREATMENT_DUMMY:
                y = list(t)
                y.append("TREATMENT_DUMMY")
                tuple_list.append(tuple(y))
            if GRF:
                y = list(t)
                y.append("GRF")
                tuple_list.append(tuple(y))
            if BCF:
                y = list(t)
                y.append("BCF")
                tuple_list.append(tuple(y))
            if TRADITIONAL:
                y = list(t)
                y.append("TRADITIONAL")
                tuple_list.append(tuple(y))

        return tuple_list

    @staticmethod
    def create_approach_list_for_single_split():
        """
        Create a list which contains the name of all approaches which should be evaluated.
        :return:
        """
        all_approaches = []

        if BCF:
            all_approaches.append("BCF")
        if CVT:
            all_approaches.append("CVT")
        if GRF:
            all_approaches.append("GRF")
        if LAIS:
            all_approaches.append("LAIS")
        if RLEARNER:
            all_approaches.append("RLEARNER")
        if SLEARNER:
            all_approaches.append("SLEARNER")
        if TRADITIONAL:
            all_approaches.append("TRADITIONAL")
        if TREATMENT_DUMMY:
            all_approaches.append("TREATMENT_DUMMY")
        if TWO_MODEL:
            all_approaches.append("TWO_MODEL")
        if URF_ED:
            all_approaches.append("URF_ED")
        if URF_KL:
            all_approaches.append("URF_KL")
        if URF_CHI:
            all_approaches.append("URF_CHI")
        if URF_DDP:
            all_approaches.append("URF_DDP")
        if URF_CTS:
            all_approaches.append("URF_CTS")
        if URF_IT:
            all_approaches.append("URF_IT")
        if URF_CIT:
            all_approaches.append("URF_CIT")
        if XLEARNER:
            all_approaches.append("XLEARNER")

        return all_approaches

    def calculate_feature_importance_mean(self, feature_importances, feature_names, dataset_name):
        """
        For each feature, calculate its mean feature importance

        :param feature_importances: Dictionary containing the feature importances
        :param feature_names: Name of the features (columns)
        :param dataset_name: Name of the data set
        :return:
        """
        # Calculate mean of feature importances
        if isinstance(feature_importances, list):
            feature_importances_sum_dict = {}

            # Run through the list of feature importances for approach and split
            for feature_importance_dict in feature_importances:
                for key, value in feature_importance_dict.items():

                    if key not in feature_importances_sum_dict:
                        feature_importances_sum_dict[key] = 0
                    feature_importances_sum_dict[key] += np.array(feature_importance_dict[key])

            for key, value in feature_importances_sum_dict.items():
                importance = value / self.cv_number_splits
                HelperPipeline.save_feature_importance(importance, feature_names, dataset_name + "_Feature_importance_{}".format(key), self.plot_save_figures, self.plot_figures)
        else:
            # Case single approach single split
            for key, value in feature_importances.items():
                importance = np.array(feature_importances[key])
                HelperPipeline.save_feature_importance(importance, feature_names, dataset_name + "_Feature_importance_{}".format(key), self.plot_save_figures, self.plot_figures)

    def set_parameters(self, n_estimators, max_depth, min_samples_leaf, min_samples_treatment, n_reg, n_jobs, normalization, honesty, random_seed):
        """
        Set the parameters for each approach
        """
        
        urf_parameters = {
            "n_estimators": n_estimators,
            "max_features": None,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_treatment": min_samples_treatment,
            "n_reg": n_reg,
            "random_state": self.random_seed,
            "n_jobs": n_jobs,
            "control_name": "c",
            "normalization": normalization,
            "honesty": honesty
        }
        
        s_learner_parameters = {
            'n_estimators': n_estimators,
            "min_samples_leaf": min_samples_leaf,
            'max_depth': max_depth,
            'max_features': self.max_features,
            'random_state': self.random_seed,
            "n_jobs": n_jobs
        }
        
        traditional_parameters = {
            'n_estimators': n_estimators,
            "min_samples_leaf": min_samples_leaf,
            'max_depth': max_depth,
            'max_features': self.max_features,
            'random_state': self.random_seed,
            "n_jobs": n_jobs
        }
        
        cvt_parameters = {
            'n_estimators': n_estimators,
            "min_samples_leaf": min_samples_leaf,
            'max_depth': max_depth,
            'max_features': self.max_features,
            'random_state': self.random_seed,
            "n_jobs": n_jobs
        }
        
        lais_parameters = {
            'n_estimators': n_estimators,
            "min_samples_leaf": min_samples_leaf,
            'max_depth': max_depth,
            'max_features': self.max_features,
            'random_state': self.random_seed,
            "n_jobs": n_jobs
        }
        
        two_model_parameters = {
            'n_estimators': n_estimators,
            "min_samples_leaf": min_samples_leaf,
            'max_depth': max_depth,
            'max_features': self.max_features,
            'random_state': self.random_seed,
            "n_jobs": n_jobs
        }
        
        x_learner_parameters = {
            'n_estimators': n_estimators,
            "min_samples_leaf": min_samples_leaf,
            'max_depth': max_depth,
            'max_features': self.max_features,
            'random_state': self.random_seed,
            "n_jobs": n_jobs
        }
        
        r_learner_parameters = {
            'n_estimators': n_estimators,
            "min_samples_leaf": min_samples_leaf,
            'max_depth': max_depth,
            'max_features': self.max_features,
            'random_state': self.random_seed,
            "n_jobs": n_jobs
        }
        
        treatment_dummy_parameters = {
            'random_state': self.random_seed,
            "n_jobs": n_jobs,
            "max_iter": 10000
        }
        
        grf_parameters = {
            "criterion": "het",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_samples": 0.45,
            "discrete_treatment": False,
            "random_state": self.random_seed,
            "n_jobs": n_jobs
        }
        
        bcf_parameters = {  # BART parameters
            "num_sweeps": 50,
            "burnin": 15,
            "num_cutpoints": 100,
            "Nmin": min_samples_leaf,
            "max_depth": max_depth,
            "parallel": True,
            "standardize_target": False,
            "set_random_seed": True,
            "random_seed": random_seed,  # Prognostic BART parameters
            "num_trees_pr": n_estimators,
            "alpha_pr": 0.95,
            "beta_pr": 2,  # Treatment BART parameters
            "num_trees_trt": n_estimators,
            "alpha_trt": 0.95,
            "beta_trt": 2,
        }

        self.parameters = {
            URF_TITLE + "_parameters": urf_parameters,
            SLEARNER_TITLE + "_parameters": s_learner_parameters,
            TRADITIONAL_TITLE + '_parameters': traditional_parameters,
            CVT_TITLE + '_parameters': cvt_parameters,
            LAIS_TITLE + '_parameters': lais_parameters,
            TWO_MODEL_TITLE + '_parameters': two_model_parameters,
            XLEARNER_TITLE + '_parameters': x_learner_parameters,
            RLEARNER_TITLE + '_parameters': r_learner_parameters,
            TREATMENT_DUMMY_TITLE + '_parameters': treatment_dummy_parameters,
            GRF_TITLE + '_parameters': grf_parameters,
            BCF_TITLE + '_parameters': bcf_parameters
        }
