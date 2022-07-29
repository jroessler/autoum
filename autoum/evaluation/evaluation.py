import logging
import os
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
from deprecated import deprecated
from matplotlib import pyplot as plt

from autoum.const.const import *

logging = logging.getLogger(__name__)


class UpliftEvaluation:
    """
    Class for evaluating uplift performance
    """

    @staticmethod
    def separate_treated_control(df_results: pd.DataFrame, treatment_col: str, response_col: str, uplift_score_col: str, treated_asc: bool = False, control_asc: bool = False):
        """
        Separate the given dataframe into two dataframes. One containing the treated samples and the other one containing the control samples.

        :param df_results: DataFrame containing treated and control samples
        :param treatment_col: Name of the treatment column
        :param response_col: Name of the response column
        :param uplift_score_col: Name of the uplift column
        :param treated_asc: True, if the treated DataFrame should be sorted by ascending order. False otherwise. Default: False
        :param control_asc: True, if the control DataFrame should be sorted by ascending order. False otherwise. Default: False
        :return: Tuple df_results_treated, df_results_control
        """
        df_results_treated = df_results.loc[df_results[treatment_col] == 1].copy()
        df_results_control = df_results.loc[df_results[treatment_col] == 0].copy()

        df_results_treated.sort_values(by=[uplift_score_col, response_col], ascending=[treated_asc, treated_asc], inplace=True)
        df_results_treated.reset_index(drop=True, inplace=True)
        df_results_control.sort_values(by=[uplift_score_col, response_col], ascending=[control_asc, control_asc], inplace=True)
        df_results_control.reset_index(drop=True, inplace=True)

        return df_results_treated, df_results_control

    @staticmethod
    def calculate_uplift(bin_treatment_responder: list,
                         bin_number_treated_samples: list,
                         bin_non_treatment_responder: list,
                         bin_number_non_treated_samples: list,
                         bin_list: list,
                         num_treated_samples):
        """
        Given the treatment responder, treatment non responder as well as the number of treated samples and number of non-treated samples for **each** bin, calculate the
        uplift score for **each** bin using Radcliffe's defintion of uplift: r_ts - ((r_cs * n_ts) / n_cs)

        :param bin_treatment_responder: List which contains the number of treatment responders for each bin
        :param bin_number_treated_samples: List which contains the number of treatment samples for each bin
        :param bin_non_treatment_responder: List which contains the number of control (non-treatment) responders for each bin
        :param bin_number_non_treated_samples: List which contains the number of control (non-treatment) responders for each bin
        :param bin_list: List with the bin values (percentiles)
        :param num_treated_samples: Number of treatment samples (overall)
        :return: Tuple bins, uplift scores (absolute) and uplift scores (relative)
        """

        bin_treatment_responder = np.array(bin_treatment_responder)
        bin_number_treated_samples = np.array(bin_number_treated_samples)
        bin_non_treatment_responder = np.array(bin_non_treatment_responder)
        bin_number_non_treated_samples = np.array(bin_number_non_treated_samples)
        bin_list = np.array(bin_list)

        # Calculate uplift
        uplift = np.around(bin_treatment_responder - np.divide(bin_non_treatment_responder * bin_number_treated_samples, bin_number_non_treated_samples,
                                                               out=np.zeros(bin_number_non_treated_samples.shape, dtype=float), where=bin_number_non_treated_samples != 0), 4)

        # Add starting point (0,0)
        bin_list = np.insert(bin_list, 0, 0)
        uplift = np.insert(uplift, 0, 0)
        uplift_pct = np.around(uplift / num_treated_samples, 4)

        return bin_list, uplift, uplift_pct

    @staticmethod
    def calculate_qini_curve(df_results: pd.DataFrame, uplift_score_col: str, bins: int = 10, treatment_col: str = 'treatment', response_col: str = 'response'):
        """
        Calculate the qini curve, once with absolute uplift values for each bin, and once with relative uplift values for each bin using Radcliffe's defintion of uplift:
        r_ts - ((r_cs * n_ts) / n_cs)

        The data set (including treatment, response and score column) is divided into treatment and control group. Both tables are sorted in descending order according to their
        score column. In order to calculate the uplift value for the first segment (e.g. decile with 10%), we have to use the first segment in the treatment and the first segment
        in the control group. Example: total_number_of_samples = 20, treated_samples = 14 and control_samples = 6 . 10% of treated_samples = 1 (rounded down) and 10% of
        control_samples = 1 (rounded up). Consequently, the calculations for the uplift value are based on one sample from the treatment group and one sample from the
        control group. Therefore, the calculations refer to the segments of treatment and control tables, respectively!

        :param df_results: DataFrame containing treated and control samples
        :param uplift_score_col: Name of the uplift column
        :param bins: Number of bins. Default: 10 (deciles)
        :param treatment_col: Name of the treatment column. Default: 'treatment'
        :param response_col: Name of the response column. Default: 'response'
        :return: Tuple bins, uplift scores (absolute) and uplift scores (relative)
        """

        df_results_treated, df_results_control = UpliftEvaluation.separate_treated_control(df_results, treatment_col, response_col, uplift_score_col)

        bin_list, r_ts, n_ts, r_cs, n_cs = UpliftEvaluation.calculate_group_share(df_results_treated, df_results_control, bins, response_col)

        return UpliftEvaluation.calculate_uplift(r_ts, n_ts, r_cs, n_cs, bin_list, df_results_treated.shape[0])

    @staticmethod
    def calculate_optimal_qini_curve(df_results: pd.DataFrame, bins: int = 10, treatment_col: str = 'treatment', response_col: str = 'response'):
        """
        Calculate optimal qini curve including negative effects. The following applies for an optimal qini curve:

        Treatment Responders > Treatment Non Responders and
        Control Non Responders > Control Responders

        :param df_results: DataFrame containing treated and control samples
        :param bins: Number of bins. Default: 10 (deciles)
        :param treatment_col: Name of the treatment column. Default: 'treatment'
        :param response_col: Name of the response column. Default: 'response'
        :return: Tuple bins, uplift scores (absolute) and uplift scores (relative)
        """

        df_results.loc[((df_results[treatment_col] == 0) & (df_results[response_col] == 0)), 'group'] = 2
        df_results.loc[((df_results[treatment_col] == 1) & (df_results[response_col] == 0)), 'group'] = 1
        df_results.loc[((df_results[treatment_col] == 0) & (df_results[response_col] == 1)), 'group'] = 0
        df_results.loc[((df_results[treatment_col] == 1) & (df_results[response_col] == 1)), 'group'] = 3

        df_results_treated = df_results.loc[df_results[treatment_col] == 1].copy()
        df_results_control = df_results.loc[df_results[treatment_col] == 0].copy()

        df_results_treated.sort_values(by=['group'], ascending=[False], inplace=True)
        df_results_treated.reset_index(drop=True, inplace=True)
        df_results_treated.drop(['group'], axis=1, inplace=True)

        df_results_control.sort_values(by=['group'], ascending=[False], inplace=True)
        df_results_control.reset_index(drop=True, inplace=True)
        df_results_control.drop(['group'], axis=1, inplace=True)

        bin_list, r_ts, n_ts, r_cs, n_cs = UpliftEvaluation.calculate_group_share(df_results_treated, df_results_control, bins, response_col)

        return UpliftEvaluation.calculate_uplift(r_ts, n_ts, r_cs, n_cs, bin_list, df_results.loc[df_results[treatment_col] == 1].shape[0])

    @staticmethod
    def calculate_random_qini_curve(bins: int, endpoint_abs: int, endpoint_pct: int, increase_value: float = 0.0):
        """
        Calculate a random qini curve

        :param bins: Number of bins
        :param endpoint_abs: Endpoint of the random curve using absolute values
        :param endpoint_pct: Endpoint of the random curve using relative values
        :param increase_value: If the endpoint is negative, we will increase each value by increase_value (because of integration issues with negative areas)
        :return: Tuple bins, uplift scores (absolute) and uplift scores (relative)
        """

        uplift_abs = np.around(np.linspace(0.0 + increase_value, endpoint_abs + increase_value, num=bins + 1), 4)
        uplift_pct = np.around(np.linspace(0.0 + increase_value, endpoint_pct + increase_value, num=bins + 1), 4)
        bin_list = np.linspace(0, 1, num=bins + 1)

        return bin_list, uplift_abs, uplift_pct

    @staticmethod
    def calculate_group_share(df_treated: pd.DataFrame, df_control: pd.DataFrame, bins: int = 10, response_col: str = 'response'):
        """
        Calculate the share of groups inherent in any data set: Treatment responder (r_ts), Treatment non-respondes (n_ts), Control responder (r_cs), Control non-responder (n_cs)

        :param df_treated: DataFrame containg the results for the treatment group
        :param df_control: DataFrame containing the results for the control group
        :param bins: Number of bins. Default: 10 (deciles)
        :param response_col: Name of the response column. Default: 'response'
        :return: bin_list (list of bins), r_ts (treatment responder), n_ts (treatment non responder), r_cs (control responder), n_cs (control non responder)
        """
        bin_list = []
        r_ts = []
        n_ts = []
        r_cs = []
        n_cs = []

        for bin_idx in np.linspace(0, 1, bins + 1):
            if bin_idx == 0.0:
                continue
            bin_idx = int(bin_idx * 1000) / float(1000)
            bin_list.append(bin_idx)
            number_of_treated_samples_bin = int(Decimal(bin_idx * df_treated.shape[0]).to_integral_value(rounding=ROUND_HALF_UP))
            number_of_control_samples_bin = int(Decimal(bin_idx * df_control.shape[0]).to_integral_value(rounding=ROUND_HALF_UP))

            if number_of_treated_samples_bin > 0 and number_of_control_samples_bin > 0:
                r_ts.append(df_treated.iloc[0:number_of_treated_samples_bin].loc[(df_treated[response_col] == 1)].shape[0])
                n_ts.append(number_of_treated_samples_bin)
                r_cs.append(df_control.iloc[0:number_of_control_samples_bin].loc[(df_control[response_col] == 1)].shape[0])
                n_cs.append(number_of_control_samples_bin)
            else:
                r_ts.append(0)
                n_ts.append(0)
                r_cs.append(0)
                n_cs.append(0)

        return bin_list, r_ts, n_ts, r_cs, n_cs

    @staticmethod
    def calculate_actual_uplift_in_bins(df_results: pd.DataFrame, bins: int = 10, treatment_col: str = 'treatment', response_col: str = 'response', absolute: bool = False) -> dict:
        """
        Calculate the uplift value for each decile and uplift modeling approach

        The uplift in each decile is defined as treatment response rate - control response rate

        :param df_results: DataFrame containing treated and control samples
        :param bins: Number of bins. Default: 10 (deciles)
        :param treatment_col: Name of the treatment column. Default: 'treatment'
        :param response_col: Name of the response column. Default: 'response'
        :param absolute: If True, return absolute values, relative values otherwise. Default: False
        :return: Dictionary containing the uplift values for each decile and approach
        """

        uplift_in_deciles = {}

        for col in df_results.drop(["response", "treatment"], axis=1).columns:
            # If the uplift modeling approach runs into an error and returns nan values; in such a case, ignore the evaluation
            if df_results[col].isnull().all():
                continue
            try:
                bin_list, uplift_abs, uplift_pct = UpliftEvaluation.calculate_qini_curve(df_results, col, bins, treatment_col, response_col)

                if absolute:
                    uplift_in_deciles.update(UpliftEvaluation.store_uplift_in_bins(uplift_abs, col + '-Uplift_'))
                else:
                    uplift_in_deciles.update(UpliftEvaluation.store_uplift_in_bins(uplift_pct, col + '-Uplift_'))

            except Exception as e:
                print("Error in calculate_uplift_in_deciles")
                print(e)
                uplift_in_deciles[col + '-Uplift_'] = np.nan

        return uplift_in_deciles

    @staticmethod
    def calculate_optimal_uplift_in_bins(df_results: pd.DataFrame, bins: int = 10, treatment_col: str = 'treatment', response_col: str = 'response', absolute: bool = False):
        """
        Calculate the optimal uplift value for each decile

        :param df_results: DataFrame (fold) which contains the treatment and control samples inclduing their responses
        :param bins: Number of bins. Default: 10 (deciles)
        :param treatment_col: Name of the treatment column. Default: 'treatment'
        :param response_col: Name of the response column. Default: 'response'
        :param absolute: If True, return absolute values, relative values otherwise. Default: False
        :return: Dictionary containing the optimal uplift values for each decile and approach
        """

        _, uplift_abs, uplift_pct = UpliftEvaluation.calculate_optimal_qini_curve(df_results, bins=bins, treatment_col=treatment_col, response_col=response_col)

        if absolute:
            return UpliftEvaluation.store_uplift_in_bins(uplift_abs, 'Optimal-Uplift_')
        else:
            return UpliftEvaluation.store_uplift_in_bins(uplift_pct, 'Optimal-Uplift_')

    @staticmethod
    def store_uplift_in_bins(uplift: np.ndarray, col_name: str):
        """
        Stores the given uplift values in a dictionary containing the column name and bin number as key and the uplift value as vaue

        :param uplift: List of uplift values
        :param col_name: Name of the key
        :return: Dictionary containing the uplift values for each bin
        """
        uplift_in_deciles = {}

        for i, decile in enumerate(uplift):
            uplift_in_deciles[col_name + str(i)] = decile

        return uplift_in_deciles

    @staticmethod
    def calculate_unscaled_qini_coefficient(df_uplift_bins: pd.DataFrame, bins: int = 10):
        """
        Calculate the unscaled qini coefficient for each approach and split (if k-fold CV):

        The unscaled qini coefficient is defined as the ratio of the actual uplift gains curve above the diagonal. Optionally divded by N^2

        The range is between 0 and infinity. The higher the score, the better the model

        Example:

        Approach A has an integral of 10, Approach B has an integral of 20, and Random Targeting has an integral of 5.
        Thus, Approach A is twice as good as Random Targeting (10/5=2), and Approach B is four times as good as Random Targeting (20/5=4).
        Thus, we can argue that the UQC expresses how much better an Approach is over Random Targeting

        :param df_uplift_bins: DataFrame containing the uplift values for each bin, approach, and fold
        :param bins: Number of bins. Default 10 (Deciles)
        :return: DataFrame which contains the uplift values (for each bin) and the unscaled qini coefficient for each approach and split
        """

        df_metrics = pd.DataFrame()

        for approach_idx in range(bins + 1, df_uplift_bins.shape[1] + 1, bins + 1):  # for each appraoch
            # Get name of uplift modeling approach
            approach_name = df_uplift_bins.columns.values[approach_idx - 11].split("-")[0]

            # Get uplift values in deciles (bins)
            df_uplift_bin_approach = df_uplift_bins.iloc[:, -(bins + 1) + approach_idx:approach_idx].copy()

            # Calculate UQC for a given split and approach
            uqc_list = []
            for row_idx in range(0, df_uplift_bins.shape[0]):  # for each split
                uplift_bins = df_uplift_bins.iloc[row_idx, -(bins + 1) + approach_idx:approach_idx].values

                endpoint = uplift_bins[-1]

                # Case 1: Endpoint is positive
                if endpoint > 0:
                    # Calculate area under random curve
                    bin_list_rand, uplift_ran, _ = UpliftEvaluation.calculate_random_qini_curve(bins, endpoint, endpoint)

                # Case 2: Endpoint is negative
                else:
                    # Calculate area under random curve
                    bin_list_rand, uplift_ran, _ = UpliftEvaluation.calculate_random_qini_curve(bins, endpoint, endpoint, abs(endpoint))

                    # Increase each value by endpoint (because of negative and positive issues when calculating an integral)
                    uplift_bins = uplift_bins + np.abs(endpoint)

                area_under_random = np.trapz(uplift_ran, x=bin_list_rand)
                area_under_curve = np.trapz(uplift_bins, x=bin_list_rand)

                uqc_list.append(round(area_under_curve / area_under_random, 4))

            df_uplift_bin_approach[approach_name + "-UQC"] = uqc_list
            df_metrics = pd.concat([df_metrics, df_uplift_bin_approach], axis=1)

        return df_metrics

    @staticmethod
    def calculate_qini_coefficient(df_metrics: pd.DataFrame, df_opt_uplift_bins: pd.DataFrame, bins: int = 10, num_columns: int = 12) -> pd.DataFrame:
        """
        Calculate the qini coefficient for each approach and split (if k-fold CV):

        The qini coefficient is defined as the ratio of the actual uplift curve above the diagonal to that of the optimum qini curve.
        Here optimal does take into account negative effects.

        The range is between -1 and 1. The higher the score, the better the model

        :param df_metrics: DataFrame containing the uplift values (for each bin) and (optionally) the unscaled qini coefficient for each approach and split
        :param df_opt_uplift_bins: DataFrame containing the optimal uplift values for each decile and split
        :param bins: Number of bins. Default 10 (Deciles)
        :param num_columns: Number of columns per approach: With UQC it's bins + 2. Without UQC it's bins +1.
        :return: DataFrame which contains the uplift values (for each bin), unscaled qini coefficient, and (optionally) the qini coefficient for each approach and split
        """

        if df_metrics.shape[0] != df_opt_uplift_bins.shape[0]:
            raise ValueError("Error in calculate_qini_coefficient. The number of rows of df_metrics and df_opt_uplift_bins is unequal")

        # Final DataFrame containing the uplift values (for each bin), unscaled qini coefficient, and the qini coefficient for each approach and split
        df_metrics_qc_extended = pd.DataFrame()

        for row_idx in range(0, df_metrics.shape[0]):  # for each split/row

            # 1. Calculate qini coefficients' denominator (globally; independent from approach)
            endpoint = df_opt_uplift_bins.iloc[row_idx].values[-1]
            uplift_opt = df_opt_uplift_bins.iloc[row_idx].values

            # Case 1: Endpoint is positive
            if endpoint > 0:
                # Calculate random curve
                bin_list_rand, uplift_rand, _ = UpliftEvaluation.calculate_random_qini_curve(bins, endpoint, endpoint)

            # Case 2: Endpoint is negative
            else:
                # Calculate random curve
                bin_list_rand, uplift_rand, _ = UpliftEvaluation.calculate_random_qini_curve(bins, endpoint, endpoint, abs(endpoint))
                uplift_opt = uplift_opt + np.abs(endpoint)

            area_under_opt = np.trapz(uplift_opt, x=bin_list_rand)
            area_under_random = np.trapz(uplift_rand, x=bin_list_rand)

            # Calculate qini coefficients' demoninator
            qini_coefficient_denominator = area_under_opt - area_under_random

            df_metrics_row = pd.DataFrame()

            # 2. Calculate qini coefficients' numerator for each approach
            for approach_idx in range(num_columns, df_metrics.shape[1] + 1, num_columns):  # for each approach
                # Get name of modeling approach
                approach_name = df_metrics.columns.values[approach_idx - num_columns].split("-")[0]

                # Depending on whether UQC values have been calculated prior to the QC values, we need to ignore these values
                if approach_idx % 12 == 0:
                    uplift_bins_split = df_metrics.iloc[row_idx, -num_columns + approach_idx:approach_idx - 1].values
                else:
                    uplift_bins_split = df_metrics.iloc[row_idx, -num_columns + approach_idx:approach_idx].values

                if endpoint < 0:
                    # Increase each value by endpoint (because of issues when calculating the integral of negative and positive areas)
                    uplift_bins_split = uplift_bins_split + np.abs(endpoint)

                area_under_curve = np.trapz(uplift_bins_split, x=bin_list_rand)

                # Calculate qini coefficient
                qini_coefficient_numerater = area_under_curve - area_under_random

                df_metrics_approach = df_metrics.iloc[[row_idx], -num_columns + approach_idx:approach_idx]
                df_metrics_approach[approach_name + "-QC"] = round(qini_coefficient_numerater / qini_coefficient_denominator, 4)

                df_metrics_row = pd.concat([df_metrics_row, df_metrics_approach], axis=1)

            df_metrics_qc_extended = pd.concat([df_metrics_qc_extended, df_metrics_row], axis=0)

        return df_metrics_qc_extended

    @staticmethod
    def calculate_mean(df_metrics: pd.DataFrame, bins: int = 10, qc: bool = False):
        """

        :param df_metrics: DataFrame containing the uplift values for each bin, approach, and fold
        :param bins: Number of bins. Default 10 (Deciles)
        :param qc: True, if the qini coefficient was calculated during the evaluation. False otherwise. Default: False
        :return: DataFrame which contains the geometric mean uplift values (for each bin), arithmetic mean unscaled qini coefficient, and (optionally) the arithmetic mean qini coefficient for each approach and split
        """
        df_metrics_mean = pd.DataFrame()

        num_cols = bins + 3 if qc else bins + 2

        for approach_idx in range(num_cols, df_metrics.shape[1] + 1, num_cols):  # for each approach

            # Get name of uplift modeling approach
            approach_name = df_metrics.columns.values[approach_idx - num_cols].split("-")[0]

            df_metrics_approach = df_metrics.iloc[:, -num_cols + approach_idx:approach_idx]

            # Calcuate arithmetic mean of uplift values for each approach and bin (decile) across the splits
            uplift_cols = [col for col in df_metrics_approach.columns if 'QC' not in col]
            arithmetic_uplift_bins = df_metrics_approach[uplift_cols].mean(axis=0).to_frame().transpose().values[0].round(4)

            # Calculate the arithmetic mean of unscaled qini coefficient values for each approach across the splits
            uqc_cols = [col for col in df_metrics_approach.columns if 'UQC' in col]
            arithmetic_uqc = np.mean(df_metrics_approach[uqc_cols].values).round(4)

            # Create a mean DataFrame
            mean_df = pd.DataFrame(arithmetic_uplift_bins).transpose()
            mean_df.columns = uplift_cols
            mean_df[approach_name + "-UQC"] = arithmetic_uqc
            # Optionally, calculate the arithmetic mean of qini coefficient values for each approach across the splits
            if qc:
                qc_cols = [col for col in df_metrics_approach.columns if '-QC' in col]
                arithmetic_qc = np.mean(df_metrics_approach[qc_cols].values).round(4)
                mean_df[approach_name + "-QC"] = arithmetic_qc

            df_metrics_mean = pd.concat([df_metrics_mean, mean_df], axis=1)

        return df_metrics_mean

    @staticmethod
    def plot_qini_curve(df_metrics: pd.DataFrame,
                        df_opt_uplift_bins: pd.DataFrame,
                        bins: int = 10,
                        plot_optimum: bool = False,
                        absolute: bool = False,
                        title: str = 'Qini Curve',
                        save_figure: bool = False,
                        path: str = None,
                        plot_grayscale: bool = False,
                        plot_uqc: bool = False,
                        show_title=False,
                        fontsize=14,
                        qc: bool = False):
        """
        Plot the qini curves for each uplift approach in the given DataFrame including random and (optionally) optimal qini curves.

        :param df_metrics: DataFrame containg the metrics such as unscaled qini coefficient or qini coefficient
        :param df_opt_uplift_bins: DataFrame containing the optimal uplift values for each bin and approach
        :param bins: Number of bins. Default: 10
        :param plot_optimum: True if the optimum curve should be plotted. False otherwise. Default: False
        :param absolute: True if the absolute uplift scores should be plotted. False otherwise. Default: False
        :param title: Title of the plot
        :param save_figure: True if the figure should be saved. False otherwise. Default: False
        :param path: Path where the figure should be saved
        :param plot_grayscale: True, if the graphics should be printed in grayscale. False otherwise. Default: False
        :param plot_uqc: True, if the legend should contain the values of the unscaled qini coefficient. False otherwise.
        :param show_title: True if the title should be shown. False otherwise. Default: False
        :param fontsize: Size of the elements in the graphic. Default: 14.
        :param qc: True, if the qini coefficient was calculated during the evaluation. False otherwise. Default: False
        """

        # Matplotlib Settings
        fig, axes = plt.subplots(figsize=(15, 10), facecolor="white")
        # Set facecolor of axes to white
        axes.set_facecolor('white')
        # Update fontsize
        plt.rcParams.update({
            'font.size': fontsize
        })
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        ## Qini Curve
        bin_list = np.linspace(0, 1, bins + 1)
        endpoint = None

        num_cols = bins + 3 if qc else bins + 2
        for approach_idx in range(num_cols, df_metrics.shape[1] + 1, num_cols):  # for each approach plot one line

            approach_name = df_metrics.columns.values[approach_idx - num_cols].split("-")[0]
            approach_label = NAMING_SCHEMA[approach_name]
            df_metrics_approach = df_metrics.iloc[:, -num_cols + approach_idx:approach_idx]

            uplift_cols = [col for col in df_metrics_approach.columns if 'QC' not in col]
            uplift_values = df_metrics_approach[uplift_cols].values[0]

            if plot_uqc:
                uqc_cols = [col for col in df_metrics_approach.columns if 'UQC' in col]
                approach_label = approach_label + " - UQC: " + str(df_metrics_approach[uqc_cols].values[0][0])

            # Plot line
            if plot_grayscale:
                axes.plot(bin_list, uplift_values, label=approach_label, color=COLOR_SCHEMA_GRAY[approach_name], linestyle=LINESTYLES[approach_name])
            else:
                axes.plot(bin_list, uplift_values, label=approach_label, color=COLOR_SCHEMA[approach_name])

            # Endpoint
            if endpoint is None:
                endpoint = uplift_values[-1]

        # Plot random curve
        bin_list_rand, uplift_rand, _ = UpliftEvaluation.calculate_random_qini_curve(bins, endpoint, endpoint)
        axes.plot(bin_list_rand, uplift_rand, label='Random', color='black', linestyle='solid')

        # Plot optimal curve
        if plot_optimum:
            axes.plot(bin_list, df_opt_uplift_bins.mean(axis=0).to_frame().transpose().values[0].round(4), label='Optimum', color='cyan')

        if absolute:
            axes.set_ylabel("Cumulative Number of Incremental Responders / Uplift (Absolute)", fontsize=fontsize)
        else:
            axes.set_ylabel("Cumulative Number of Incremental Responders / Uplift (Relative)", fontsize=fontsize)

        axes.set_xlabel("Fraction of data targeted (in %)", fontsize=fontsize)

        # Sort both labels and handles by labels
        handles, labels = axes.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        axes.legend(handles, labels, facecolor="white")

        plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"])

        if show_title:
            axes.set_title(title)

        if save_figure:
            if not os.path.exists(path):
                os.makedirs(path)
            now = datetime.now()
            now_date = now.date()
            plt.savefig(path + title.replace(" ", "_") + "_" + str(now_date) + ".png", facecolor=fig.get_facecolor(), transparent=True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_gains_chart(df_results: pd.DataFrame):
        """
        Evaluate the results of the given dataset by calculating the cumulative sum. (Gains Chart)

        :param df_results: Pandas Dataframe. Each column is a score calculated by an algorithm e.g. Direct Uplift etc.
        """

        fig, axes = plt.subplots()

        # Random Line
        axes.plot((0, df_results.shape[0]), (0, df_results.respomse.sum()), label="Random selection", linestyle='dashed')

        for col in df_results.drop(["response", "treatment"], axis=1).columns:
            # Calculate cumulative sum for each column
            cumulative = np.cumsum(df_results.sort_values(by=col, ascending=False)["response"].tolist())
            # auc_score = auc(np.arange(0, 41650), cumulative)
            axes.plot(cumulative, label=col)

        axes.legend()
        axes.set_title("Gains Chart")
        plt.show()

    @staticmethod
    @deprecated(version='1.0', reason="This function is only implemented for the sake of completeness. Use calculate_qini_curve instead")
    def calculate_qini_curve2(df_results: pd.DataFrame, uplift_score_col: str, bins: int = 10, treatment_col: str = 'treatment', response_col: str = 'response'):
        """
        Calculate absolute and relative uplift values for each bin using Radcliffe's defintion of uplift: r_ts - ((r_cs * n_ts) / n_cs)

        The data set (including treatment, response and score column) is not split. The table is sorted in descending order according to the score column. The calculation for
        the uplift value of the first segment (e.g. decile with 10%) refers to the whole table. Example: total_number_of_samples = 20, treated_samples = 14, control_samples = 6.
        10% of the whole table = 2. Consequently, the calculations of the uplift values refer to the first two values of the table, regardless of whether they belong to the
        treatment or control group.

        :param df_results: DataFrame containing treated and control samples
        :param uplift_score_col: Name of the uplift column
        :param bins: Number of bins. Default: 10 (deciles)
        :param treatment_col: Name of the treatment column. Default: 'treatment'
        :param response_col: Name of the response column. Default: 'response'
        :return: Tuple bins, uplift scores (absolute) and uplift scores (relative)
        """

        df_results.sort_values(by=[uplift_score_col, response_col], ascending=[False, False], inplace=True)
        df_results.reset_index(drop=True, inplace=True)

        bin_list = []
        r_ts = []
        n_ts = []
        r_cs = []
        n_cs = []

        for bin_idx in np.linspace(0, 1, bins + 1):
            if bin_idx == 0.0:
                continue
            bin_idx = int(bin_idx * 1000) / float(1000)
            bin_list.append(bin_idx)
            number_of_samples_bin = int(Decimal(bin_idx * df_results.shape[0]).to_integral_value(rounding=ROUND_HALF_UP))

            if number_of_samples_bin > 0:
                df_results_treated = df_results.iloc[0:number_of_samples_bin].loc[df_results[treatment_col] == 1].copy()
                df_results_control = df_results.iloc[0:number_of_samples_bin].loc[df_results[treatment_col] == 0].copy()

                r_ts.append(df_results_treated.loc[(df_results_treated[response_col] == 1)].shape[0])
                n_ts.append(df_results_treated.shape[0])
                r_cs.append(df_results_control.loc[(df_results_control[response_col] == 1)].shape[0])
                n_cs.append(df_results_control.shape[0])
            else:
                r_ts.append(0)
                n_ts.append(0)
                r_cs.append(0)
                n_cs.append(0)

        return UpliftEvaluation.calculate_uplift(r_ts, n_ts, r_cs, n_cs, bin_list, df_results.loc[df_results[treatment_col] == 1].shape[0])

    @staticmethod
    @deprecated(version='1.0', reason="This function is only implemented for the sake of completeness. Use calculate_optimal_qini_curve instead")
    def calculate_optimal_qini_curve2(df_results: pd.DataFrame, bins: int, treatment_col: str, response_col: str):
        """
        Calculate optimal qini curve including negative effects. The following applies for an optimal qini curve:

        Treatment Responders > Treatment Non Responders > Control Non Responders > Control Responders

        :param df_results: DataFrame containing treated and control samples
        :param bins: Number of bins
        :param treatment_col: Name of the treatment column
        :param response_col: Name of the response column
        :return: Tuple bins, uplift scores (absolute) and uplift scores (relative)
        """

        df_results.loc[((df_results[treatment_col] == 0) & (df_results[response_col] == 0)), 'group'] = 2
        df_results.loc[((df_results[treatment_col] == 1) & (df_results[response_col] == 0)), 'group'] = 1
        df_results.loc[((df_results[treatment_col] == 0) & (df_results[response_col] == 1)), 'group'] = 0
        df_results.loc[((df_results[treatment_col] == 1) & (df_results[response_col] == 1)), 'group'] = 3

        df_results.sort_values(by=['group'], ascending=[False], inplace=True)
        df_results.reset_index(drop=True, inplace=True)
        df_results.drop(['group'], axis=1, inplace=True)

        bin_list = []
        r_ts = []
        n_ts = []
        r_cs = []
        n_cs = []

        for bin_idx in np.linspace(0, 1, bins + 1):
            if bin_idx == 0.0:
                continue
            bin_idx = int(bin_idx * 1000) / float(1000)
            bin_list.append(bin_idx)
            number_of_samples_bin = int(Decimal(bin_idx * df_results.shape[0]).to_integral_value(rounding=ROUND_HALF_UP))

            if number_of_samples_bin > 0:
                df_results_treated = df_results.iloc[0:number_of_samples_bin].loc[df_results[treatment_col] == 1].copy()
                df_results_control = df_results.iloc[0:number_of_samples_bin].loc[df_results[treatment_col] == 0].copy()

                r_ts.append(df_results_treated.loc[(df_results_treated[response_col] == 1)].shape[0])
                n_ts.append(df_results_treated.shape[0])
                r_cs.append(df_results_control.loc[(df_results_control[response_col] == 1)].shape[0])
                n_cs.append(df_results_control.shape[0])
            else:
                r_ts.append(0)
                n_ts.append(0)
                r_cs.append(0)
                n_cs.append(0)

        return UpliftEvaluation.calculate_uplift(r_ts, n_ts, r_cs, n_cs, bin_list, df_results.loc[df_results[treatment_col] == 1].shape[0])
