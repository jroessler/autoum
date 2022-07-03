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


class Hillstrom:

    def __init__(self):

        ## Define paths
        load_dotenv()
        self.parent_folder = os.getenv("ROOT_FOLDER")
        self.data_folder = self.parent_folder + "data/"
        self.hillstrom_path_folder = self.data_folder + "hillstrom-email/"
        self.hillstrom_original_path = self.hillstrom_path_folder + "hillstrom.csv"
        self.hillstrom_visit_path = self.hillstrom_path_folder + "Hillstrom_Email_visit.csv"
        self.hillstrom_w_visit_path = self.hillstrom_path_folder + "Hillstrom_Email_w_visit.csv"
        self.hillstrom_m_visit_path = self.hillstrom_path_folder + "Hillstrom_Email_m_visit.csv"
        self.hillstrom_conversion_path = self.hillstrom_path_folder + "Hillstrom_Email_conversion.csv"
        self.hillstrom_w_conversion_path = self.hillstrom_path_folder + "Hillstrom_Email_w_conversion.csv"
        self.hillstrom_m_conversion_path = self.hillstrom_path_folder + "Hillstrom_Email_m_conversion.csv"

    def prep(self, only_women=False, only_men=False, visit_T=True):
        """
        Prepare the Hillstrom dataset and store the csv files in the filesystem.

        1. Resample both groups such that they have the same amount of items (# treated = # controlled)
        2. Apply LabelEncoder on categorical columns
        3. Create target column with a binary value

        :param only_men: True, if only the men treatment should be considered. False otherwise.
        :param only_women: True, if only thewomen treatment should be considered. False otherwise.
        :param visit_T: True (default) if the feature "visit" shall be used as response variable.
            False if the feature "conversion" shall be the response variable. While visit indicates just a small response,
            conversion indicates that the customer actually bought products.

        """
        data = pd.read_csv(self.hillstrom_original_path)

        # 1. Create treatment column
        if only_women:
            data = data.loc[(data.segment != "Mens E-Mail")].copy()
            data["treatment"] = [0 if x == "No E-Mail" else 1 for x in data.segment]
        elif only_men:
            data = data.loc[(data.segment != "Womens E-Mail")].copy()
            data["treatment"] = [0 if x == "No E-Mail" else 1 for x in data.segment]
        else:
            data["treatment"] = [0 if x == "No E-Mail" else 1 for x in data.segment]

        data.drop((["segment"]), axis=1, inplace=True)

        ## 2. Apply LabelEncoder on categorical columns
        le = preprocessing.LabelEncoder()
        oe = preprocessing.OrdinalEncoder()
        data.history_segment = oe.fit_transform(data.history_segment.to_numpy().reshape(-1, 1)).astype('int32')
        data.zip_code = le.fit_transform(data.zip_code)
        data.channel = le.fit_transform(data.channel)

        ## 3. Create target column with a binary value
        if visit_T:
            data["response"] = [1 if x > 0.0 else 0 for x in data.visit]
        else:
            data["response"] = [1 if x > 0.0 else 0 for x in data.conversion]

        data.drop((["spend"]), axis=1, inplace=True)
        data.drop((["conversion"]), axis=1, inplace=True)
        data.drop((["visit"]), axis=1, inplace=True)

        # 4. Remove duplicates
        data.drop_duplicates(inplace=True, ignore_index=True)
        data.reset_index(inplace=True, drop=True)

        if only_women:
            if visit_T:
                save_path = self.hillstrom_w_visit_path
            else:
                save_path = self.hillstrom_w_conversion_path

        elif only_men:
            if visit_T:
                save_path = self.hillstrom_m_visit_path
            else:
                save_path = self.hillstrom_m_conversion_path

        else:
            if visit_T:
                save_path = self.hillstrom_visit_path
            else:
                save_path = self.hillstrom_conversion_path

        data.to_csv(save_path, index=False)


if __name__ == '__main__':
    hillstrom = Hillstrom()
    hillstrom.prep(only_women=False, only_men=False, visit_T=False)
    hillstrom.prep(only_women=False, only_men=True, visit_T=True)
    hillstrom.prep(only_women=True, only_men=False, visit_T=False)
    hillstrom.prep(only_women=False, only_men=False, visit_T=True)
    hillstrom.prep(only_women=False, only_men=True, visit_T=False)
    hillstrom.prep(only_women=True, only_men=False, visit_T=True)
    eda(hillstrom.hillstrom_visit_path)
    eda(hillstrom.hillstrom_w_visit_path)
    eda(hillstrom.hillstrom_m_visit_path)
    eda(hillstrom.hillstrom_conversion_path)
    eda(hillstrom.hillstrom_w_conversion_path)
    eda(hillstrom.hillstrom_m_conversion_path)
