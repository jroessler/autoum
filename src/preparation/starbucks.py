import os
import sys

import pandas as pd
from dotenv import load_dotenv
from sklearn import preprocessing

load_dotenv()
root = os.getenv("ROOT_FOLDER")
sys.path.append(root + "src/")

from preparation.helper.helper_preparation import eda

pd.set_option('display.max_columns', 100)


class Starbucks:

    def __init__(self):
        # Define paths
        load_dotenv()
        self.parent_folder = os.getenv("ROOT_FOLDER")
        self.data_folder = self.parent_folder + "data/"
        self.starbucks_folder = self.data_folder + "starbucks/"
        self.starbucks_train_path = self.starbucks_folder + "Training.csv"
        self.starbucks_test_path = self.starbucks_folder + "Test.csv"
        self.starbucks_path = self.starbucks_folder + "starbucks.csv"

    def prep(self):
        """
        1. Rename columns
        2. Encode treatment column
        3. Remove duplicates
        """

        data_train = pd.read_csv(self.starbucks_train_path)
        data_test = pd.read_csv(self.starbucks_test_path)

        # 1. Rename "conversion" column to "response"
        data_train.rename(columns={
            "purchase": "response",
            "Promotion": "treatment"
            }, inplace=True)
        data_test.rename(columns={
            "purchase": "response",
            "Promotion": "treatment"
            }, inplace=True)

        # 2. Encode treatment column
        le = preprocessing.LabelEncoder()
        data_train.treatment = le.fit_transform(data_train.treatment)
        data_test.treatment = le.fit_transform(data_test.treatment)

        data = pd.concat([data_train, data_test]).sample(frac=1)
        data.drop(["ID"], axis=1, inplace=True)

        le = preprocessing.LabelEncoder()
        data["V1"] = le.fit_transform(data["V1"])
        data["V5"] = le.fit_transform(data["V5"])
        data["V6"] = le.fit_transform(data["V6"])

        # 3. Remove duplicates
        data.drop_duplicates(inplace=True, ignore_index=True)
        data.reset_index(inplace=True, drop=True)

        data.to_csv(self.starbucks_path, index=False)


if __name__ == '__main__':
    starbucks = Starbucks()
    starbucks.prep()
    eda(starbucks.starbucks_path)
