import pandas as pd
from sklearn import preprocessing
from os import path

pd.set_option('display.max_columns', 100)


class Starbucks:

    def __init__(self, path_folder: str):
        # Define paths
        self.starbucks_train_path = path_folder + "Training.csv"
        self.starbucks_test_path = path_folder + "Test.csv"
        self.starbucks_path = path_folder + "starbucks.csv"

    def prep(self):
        """
        1. Rename columns
        2. Encode treatment column
        3. Remove duplicates
        """
        if path.exists(self.starbucks_path):
            return pd.read_csv(self.starbucks_path)

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

        return data
