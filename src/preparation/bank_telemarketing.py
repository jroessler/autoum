import os
import sys

import numpy as np
from sklearn import preprocessing

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
root = os.getenv("ROOT_FOLDER")
sys.path.append(root + "src/")

from preparation.helper.helper_preparation import eda

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 320)


class Bank_Telemarketing:

    def __init__(self):
        ## Define paths
        load_dotenv()
        self.parent_folder = os.getenv("ROOT_FOLDER")
        self.data_folder = self.parent_folder + "data/"
        self.bank_telemarketing_path = self.data_folder + "bank-telemarketing/"
        self.bank_additional_original_path = self.bank_telemarketing_path + "bank-additional-full.csv"
        self.bank_additional_path_1 = self.bank_telemarketing_path + "Bank-Telemarketing_1.csv"
        self.bank_additional_path_2 = self.bank_telemarketing_path + "Bank-Telemarketing_2.csv"

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

    def eda_extended(self):
        """
        Explorative Data Analysis
        :return:
        """
        data = pd.read_csv(self.bank_additional_original_path, sep=';')
        # print(data.pdays.value_counts())
        # print(data.campaign.value_counts())

        # Check for NaN values
        print(data.isnull().values.any())

        for col in data.columns:
            print(data[col].unique())
            print(data[col].dtype)

        # 1. Apply LabelEncoder on categorical columns
        le = preprocessing.LabelEncoder()
        data.job = le.fit_transform(data.job)
        data.marital = le.fit_transform(data.marital)
        data.education = le.fit_transform(data.education)
        data.default = le.fit_transform(data.default)
        data.housing = le.fit_transform(data.housing)
        data.loan = le.fit_transform(data.loan)
        data.month = le.fit_transform(data.month)
        data.day_of_week = le.fit_transform(data.day_of_week)
        data.pdays = le.fit_transform(data.pdays)
        data.poutcome = le.fit_transform(data.poutcome)

        # Cellular vs Telephone
        data["treatment"] = [0 if x == "telephone" else 1 for x in data.contact]
        data.drop((["contact"]), axis=1, inplace=True)

        # # Treated = Contacted via cellular and not contacted in last campaign
        # data["contacted"] = [0 if x == "telephone" else 1 for x in data.contact]
        # data["contacted_last_campaign"] = [1 if x == 1 else 0 for x in data.poutcome]
        # data["treatment"] =  data["contacted"] * data["contacted_last_campaign"]
        #
        # data.drop((["contact"]), axis=1, inplace=True)
        # data.drop((["contacted"]), axis=1, inplace=True)
        # data.drop((["contacted_last_campaign"]), axis=1, inplace=True)

        # # For customers who were only contacted once, it is assumed that they were not influenced by the campaign
        # data["treatment"] = [0 if x == 1 else 1 for x in data.campaign]
        # data.drop((["campaign"]), axis=1, inplace=True)
        #
        # # Contacted in last campaign
        # data["treatment"] = [0 if x == "nonexistent"  else 1 for x in data.poutcome]
        # data.drop((["poutcome"]), axis=1, inplace=True)

        # 3. Create target column with a binary value
        data["response"] = [1 if x == "yes" else 0 for x in data.y]
        data.drop((["y"]), axis=1, inplace=True)
        # data.drop((["duration"]), axis=1, inplace=True)

        print("***** Matrix *****")
        print(pd.crosstab(data['response'], data['treatment'], margins=True))
        print()

        data_treated = data.loc[data.treatment == 1]
        data_control = data.loc[data.treatment == 0]

        print("Number of rows: {} and number of features: {}".format(data.shape[0], data.shape[1]))
        print("Number of treated samples: {}".format(data_treated.shape[0]))
        print("Number of control samples: {}".format(data_control.shape[0]))
        print("Treated response rate: {}".format((data_treated.loc[data_treated.response == 1].shape[0] / data_treated.shape[0])))
        print("Control response rate: {}".format(data_control.loc[data_control.response == 1].shape[0] / data_control.shape[0]))

        # print(data_treated.describe())
        print("Summary statistics:")
        print("Mean:")
        print(data.groupby("treatment").mean())
        print("Variance:")
        print(data.groupby("treatment").var())


if __name__ == '__main__':
    tele = Bank_Telemarketing()
    tele.eda_extended()
    tele.prep_contact()
    tele.prep_outcome()
    eda(tele.bank_additional_path_1)
    eda(tele.bank_additional_path_2)
