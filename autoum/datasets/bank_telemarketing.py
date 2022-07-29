from os import path

import numpy as np
import pandas as pd
from sklearn import preprocessing

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 320)


class Bank_Telemarketing:

    def __init__(self, path_folder: str):
        ## Define paths
        self.bank_additional_original_path = path_folder + "bank-additional-full.csv"
        self.bank_additional_path_1 = path_folder + "Bank-Telemarketing_1.csv"
        self.bank_additional_path_2 = path_folder + "Bank-Telemarketing_2.csv"

    def prep_contact(self):
        """
        Prepare the bank_telemarketing dataset and store the csv files in the filesystem. Here only the contact column
        is used as the basis for creating the treatment column.

        Note: The duration column highly affects the output target (e.g., if duration=0 then y='no').
        Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known.
        Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to
        have a realistic predictive model.

        1. Apply LabelEncoder on categorical columns
        2. Encode treatment and resample both groups such that they have the same amount of items (# treated = # controlled)
        3. Create target column with a binary value
        4. Drop duplicates
        """

        if path.exists(self.bank_additional_path_1):
            return pd.read_csv(self.bank_additional_path_1)

        data = pd.read_csv(self.bank_additional_original_path, sep=';')

        # 1. Apply LabelEncoder on categorical columns
        # Find all columns to encode. All colunms of data type object will be encoded. All columns that should not be
        # encoded, must have a dtype different than object. Thus before using this method, please be sure to convert all
        # columns to the correct data type.
        encode_columns = []

        for col in data:
            if data[col].dtype == np.object:
                encode_columns.append(col)

        # Exclude treatment and control variable
        encode_columns.remove("contact")
        encode_columns.remove("y")

        le = preprocessing.LabelEncoder()
        for column in encode_columns:
            data[column] = le.fit_transform(data[column])

        # 2. Encode treatment
        # Cellular vs Telephone
        data["treatment"] = [0 if x == "telephone" else 1 for x in data.contact]
        data.drop((["contact"]), axis=1, inplace=True)

        # 3. Create target column with a binary value
        data["response"] = [1 if x == "yes" else 0 for x in data.y]
        data.drop((["y"]), axis=1, inplace=True)
        data.drop((["duration"]), axis=1, inplace=True)

        # 4. Remove duplicates
        data.drop_duplicates(inplace=True, ignore_index=True)
        data.reset_index(inplace=True, drop=True)

        data.to_csv(self.bank_additional_path_1, index=False)

        return data

    def prep_outcome(self):
        """
        Prepare the bank_telemarketing dataset and store the csv files in the filesystem. Here the contact column and
        the outcome column are used as the basis for creating the treatment column.
        Treatment = contacted cellular & not contacted in last campaign

        Note: The duration column highly affects the output target (e.g., if duration=0 then y='no').
        Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known.
        Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to
        have a realistic predictive model.

        1. Apply LabelEncoder on categorical columns
        2. Encode treatment and resample both groups such that they have the same amount of items (# treated = # controlled)
        3. Create target column with a binary value
        4. Drop duplicates
        """

        if path.exists(self.bank_additional_path_2):
            return pd.read_csv(self.bank_additional_path_2)

        data = pd.read_csv(self.bank_additional_original_path, sep=';')

        # 1. Apply LabelEncoder on categorical columns
        # Find all columns to encode. All colunms of data type object will be encoded. All columns that should not be
        # encoded, must have a dtype different than object. Thus before using this method, please be sure to convert all
        # columns to the correct data type.
        encode_columns = []

        for col in data:
            if data[col].dtype == np.object:
                encode_columns.append(col)

        # Exclude treatment and control variable
        encode_columns.remove("contact")
        encode_columns.remove("y")

        le = preprocessing.LabelEncoder()
        for column in encode_columns:
            data[column] = le.fit_transform(data[column])

        # 2. Encode treatment
        # Treated = Contacted via cellular and not contacted in last campaign
        data["contacted"] = [0 if x == "telephone" else 1 for x in data.contact]
        data["contacted_last_campaign"] = [1 if x == 1 else 0 for x in data.poutcome]
        data["treatment"] = data["contacted"] * data["contacted_last_campaign"]

        data.drop((["contact"]), axis=1, inplace=True)
        data.drop((["contacted"]), axis=1, inplace=True)
        data.drop((["contacted_last_campaign"]), axis=1, inplace=True)

        # 3. Create target column with a binary value
        data["response"] = [1 if x == "yes" else 0 for x in data.y]
        data.drop((["y"]), axis=1, inplace=True)
        data.drop((["duration"]), axis=1, inplace=True)

        # 4. Remove duplicates
        data.drop_duplicates(inplace=True, ignore_index=True)
        data.reset_index(inplace=True, drop=True)

        data.to_csv(self.bank_additional_path_2, index=False)

        return data
