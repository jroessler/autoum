import os
import sys

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
root = os.getenv("ROOT_FOLDER")
sys.path.append(root + "src/")

from preparation.helper.helper_preparation import eda

pd.set_option('display.max_columns', 100)


class CriteoV2:

    def __init__(self):
        # Define paths

        load_dotenv()
        self.parent_folder = os.getenv("ROOT_FOLDER")
        self.data_folder = self.parent_folder + "data/"
        self.criteo_v2_path_folder = self.data_folder + "criteo-marketing-v2/"
        self.criteo_v2_path_original = self.criteo_v2_path_folder + "criteo-uplift-v2.1.csv"
        self.criteo_v2_path = self.criteo_v2_path_folder + "criteo_uplift_v2.csv"
        self.criteo_v2_path_resampled = self.criteo_v2_path_folder + "criteo_uplift_v2_resampled.csv"

    def prep(self, resample=False):
        """
        Prepare the Criteo_V2 dataset and store the csv files in the filesystem.

        1. Rename columns
        2. Downsample majority class (treatment group)
        3. Delete unnecessary columns (visit, exposure)
        4. Drop duplicates
        """

        data = pd.read_csv(self.criteo_v2_path_original)

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
            data.to_csv(self.criteo_v2_path_resampled, index=False)
        else:
            data.to_csv(self.criteo_v2_path, index=False)


if __name__ == '__main__':
    criteo_v2 = CriteoV2()
    criteo_v2.prep(resample=True)
    criteo_v2.prep(resample=False)
    eda(criteo_v2.criteo_v2_path)
