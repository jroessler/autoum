from os import path

import pandas as pd

pd.set_option('display.max_columns', 100)


class Criteo:

    def __init__(self, path_folder: str):
        # Define paths
        self.criteo_path_original = path_folder + "criteo-uplift.csv"
        self.criteo_path = path_folder + "criteo_uplift.csv"
        # Resample
        self.criteo_path_resampled = path_folder + "criteo_uplift_resampled.csv"

    def prep(self, resample: bool=False):
        """
        Prepare the Criteo dataset and store the csv files in the filesystem.

        1. Rename columns
        2. Downsample majority class (treatment group)
        3. Delete unnecessary columns (visit, exposure)
        4. Drop duplicates

        :param resample: True if the dataset should be downsampled. False otherwise. Default: False.
        """

        if resample:
            if path.exists(self.criteo_path):
                return pd.read_csv(self.criteo_path)
        else:
            if path.exists(self.criteo_path_resampled):
                return pd.read_csv(self.criteo_path_resampled)

        data = pd.read_csv(self.criteo_path_original)

        # 1. Rename "conversion" column to "response"
        data.rename(columns={
            "conversion": "response"
        }, inplace=True)

        # 2. Downsample the treatment group
        if resample:
            data_no_treatment = data.loc[data.treatment == 0]
            data_treatment = data.loc[data.treatment == 1]

            frac = round(data_no_treatment.shape[0] / data_treatment.shape[0], 3)

            data_treatment = data_treatment.sample(frac=frac)

            data = pd.concat([data_treatment, data_no_treatment]).sample(frac=1)

        # 3. Delete unnecessary columns
        data.drop(["visit", "exposure"], axis=1, inplace=True)

        # 4. Remove duplicates
        data.drop_duplicates(inplace=True, ignore_index=True)
        data.reset_index(inplace=True, drop=True)

        if resample:
            data.to_csv(self.criteo_path_resampled, index=False)
        else:
            data.to_csv(self.criteo_path, index=False)

        return data
