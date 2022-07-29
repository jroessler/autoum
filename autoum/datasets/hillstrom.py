from os import path

import pandas as pd
from sklearn import preprocessing

pd.set_option('display.max_columns', 100)


class Hillstrom:

    def __init__(self, path_folder: str):

        self.hillstrom_original_path = path_folder + "hillstrom.csv"
        self.hillstrom_visit_path = path_folder + "Hillstrom_Email_visit.csv"
        self.hillstrom_w_visit_path = path_folder + "Hillstrom_Email_w_visit.csv"
        self.hillstrom_m_visit_path = path_folder + "Hillstrom_Email_m_visit.csv"
        self.hillstrom_conversion_path = path_folder + "Hillstrom_Email_conversion.csv"
        self.hillstrom_w_conversion_path = path_folder + "Hillstrom_Email_w_conversion.csv"
        self.hillstrom_m_conversion_path = path_folder + "Hillstrom_Email_m_conversion.csv"

    def prep(self, only_women: bool=False, only_men: bool=False, visit: bool=True):
        """
        Prepare the Hillstrom dataset and store the csv files in the filesystem.

        1. Resample both groups such that they have the same amount of items (# treated = # controlled)
        2. Apply LabelEncoder on categorical columns
        3. Create target column with a binary value

        :param only_women: True, if only th ewomen treatment should be considered. False otherwise.
        :param only_men: True, if only the men treatment should be considered. False otherwise.
        :param visit: True (default) if the feature "visit" shall be used as response variable.
            False if the feature "conversion" shall be the response variable. While visit indicates just a small response,
            conversion indicates that the customer actually bought products.

        """
        if not only_women and not only_men and not visit:
            if path.exists(self.hillstrom_conversion_path):
                return pd.read_csv(self.hillstrom_conversion_path)
        elif not only_women and not only_men and visit:
            if path.exists(self.hillstrom_visit_path):
                return pd.read_csv(self.hillstrom_visit_path)
        elif only_women and not only_men and visit:
            if path.exists(self.hillstrom_w_visit_path):
                return pd.read_csv(self.hillstrom_w_visit_path)
        elif only_women and not only_men and not visit:
            if path.exists(self.hillstrom_w_conversion_path):
                return pd.read_csv(self.hillstrom_w_conversion_path)
        elif not only_women and only_men and visit:
            if path.exists(self.hillstrom_m_visit_path):
                return pd.read_csv(self.hillstrom_m_visit_path)
        elif not only_women and only_men and not visit:
            if path.exists(self.hillstrom_m_conversion_path):
                return pd.read_csv(self.hillstrom_m_conversion_path)

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
        if visit:
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
            if visit:
                save_path = self.hillstrom_w_visit_path
            else:
                save_path = self.hillstrom_w_conversion_path

        elif only_men:
            if visit:
                save_path = self.hillstrom_m_visit_path
            else:
                save_path = self.hillstrom_m_conversion_path

        else:
            if visit:
                save_path = self.hillstrom_visit_path
            else:
                save_path = self.hillstrom_conversion_path

        data.to_csv(save_path, index=False)

        return data
