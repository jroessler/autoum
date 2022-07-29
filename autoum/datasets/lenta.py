from os import path

import pandas as pd
from sklearn import preprocessing

pd.set_option('display.max_columns', 100)


class Lenta:

    def __init__(self, path_folder: str):
        # Define paths

        self.lena_path_original = path_folder + "lenta_dataset.csv"
        self.lenta_path = path_folder + "lenta-dataset.csv"

    def prep(self):
        """
        Prepare the Lenta dataset and store the csv files in the filesystem.

        1. Rename columns
        2. Delete columns with noo many missing values (as this data set has a lot of missing values)
        3. Remove rows containing any missing values

        Note that we reduce the number of rows from 687.029 to 176.065. Further, we reduce the number of columns from 195 to 110
        :return:
        """

        if path.exists(self.lenta_path):
            return pd.read_csv(self.lenta_path)

        data = pd.read_csv(self.lena_path_original)

        # 1. Rename "response_att" column to "response"
        data.rename(columns={
            "response_att": "response",
            "group": "treatment"
            }, inplace=True)

        # 2. Remove columns with too many missing values
        threshold = 0.8  # 80% of the values (for each column) should NOT be missing
        data.dropna(thresh=int(threshold * data.shape[0]), axis=1, inplace=True)

        # 3. Remove rows containin any missing values
        data.dropna(inplace=True)
        data.reset_index(inplace=True, drop=True)

        # 4. Transform gender variable
        le = preprocessing.LabelEncoder()
        data.gender = le.fit_transform(data.gender)

        # 4. Transform treatment variable
        data["treatment"] = [0 if x == "control" else 1 for x in data.treatment]

        data.to_csv(self.lenta_path, index=False)

        return data
