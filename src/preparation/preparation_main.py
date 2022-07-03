import gzip
import os
import shutil
import sys
import tarfile
from zipfile import ZipFile

from dotenv import load_dotenv

load_dotenv()
root = os.getenv("ROOT_FOLDER")
sys.path.append(root + "src/")

from preparation.bank_telemarketing import Bank_Telemarketing
from preparation.criteo import Criteo
from preparation.hillstrom import Hillstrom
from preparation.starbucks import Starbucks
from preparation.socialpressure import SocialPressure
from preparation.lenta import Lenta
from preparation.criteo_v2 import CriteoV2


class Preparation():

    def __init__(self):
        # Define paths
        self.parent_folder = "../../"
        self.data_folder = self.parent_folder + "data/"

        self.bank_telemarketing_path = self.data_folder + "bank-telemarketing/"
        self.bank_telemarketing_zip_path = self.bank_telemarketing_path + "bank-additional-full.csv.zip"
        self.criteo_path = self.data_folder + "criteo-marketing/"
        self.criteo_path_zip_path = self.criteo_path + "criteo-uplift.csv.gz"
        self.hillstrom_path = self.data_folder + "hillstrom-email/"
        self.hillstrom_path_zip_path = self.hillstrom_path + "hillstrom.csv.zip"
        self.starbucks_path = self.data_folder + "starbucks/"
        self.starbucks_zip_path = self.starbucks_path + "data.tar.gz"
        self.social_pressure_path = self.data_folder + "social-pressure/"
        self.social_pressure_zip_path = self.social_pressure_path + "data.zip"
        self.lenta_path_folder = self.data_folder + "lenta/"
        self.lenta_zip_path = self.lenta_path_folder + "lenta_dataset.csv.gz"
        self.criteo_v2_path_folder = self.data_folder + "criteo-marketing-v2/"
        self.criteo_v2_zip_path = self.criteo_v2_path_folder + "criteo-uplift-v2.1.csv.gz"

    def unpack_compressed_files(self):
        """
        Unpacks all public data sets that are available in the data folder.

        """

        # Unpack bank_telemarketing
        with ZipFile(self.bank_telemarketing_zip_path, 'r') as zip_file:
            zip_file.extractall(self.bank_telemarketing_path)

        print('Unpacked ' + self.bank_telemarketing_zip_path + ' successfully')

        # Unpack criteo
        with gzip.open(self.criteo_path_zip_path, 'rb') as f_in:
            with open(self.criteo_path_zip_path[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        print('Unpacked ' + self.criteo_path_zip_path + ' successfully')

        # Unpack hillstrom
        with ZipFile(self.hillstrom_path_zip_path, 'r') as zip_file:
            zip_file.extractall(self.hillstrom_path)

        print('Unpacked ' + self.hillstrom_path_zip_path + ' successfully')

        # Unpack starbucks
        with tarfile.open(self.starbucks_zip_path) as my_tar:
            my_tar.extractall(self.starbucks_path)

        print('Unpacked ' + self.starbucks_zip_path + ' successfully')

        # Unpack social pressure
        with ZipFile(self.social_pressure_zip_path, 'r') as zip_file:
            zip_file.extractall(self.social_pressure_path)

        print('Unpacked ' + self.social_pressure_zip_path + ' successfully')

        # Unpack lenta
        with gzip.open(self.lenta_zip_path, 'rb') as f_in:
            with open(self.lenta_zip_path[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        print('Unpacked ' + self.lenta_zip_path + ' successfully')

        # Unpack criteo_v2
        with gzip.open(self.criteo_v2_zip_path, 'rb') as f_in:
            with open(self.criteo_v2_zip_path[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        print('Unpacked ' + self.criteo_v2_zip_path + ' successfully')

    def prepare_all_data(self):
        """
        Calls all preparation methods needed to create data sets that can be used for uplift modelling.
        """

        # Prepare bank_telemarketing data
        tele = Bank_Telemarketing()
        tele.prep_contact()
        tele.prep_outcome()

        # Prepare criteo data
        criteo = Criteo()
        criteo.prep(resample=False)
        criteo.prep(resample=True)

        # Prepare hillstrom data
        hillstrom = Hillstrom()
        hillstrom.prep(only_women=False, only_men=False, visit_T=False)
        hillstrom.prep(only_women=False, only_men=True, visit_T=True)
        hillstrom.prep(only_women=True, only_men=False, visit_T=False)
        hillstrom.prep(only_women=False, only_men=False, visit_T=True)
        hillstrom.prep(only_women=False, only_men=True, visit_T=False)
        hillstrom.prep(only_women=True, only_men=False, visit_T=True)

        # Prepare starbucks data
        starbucks = Starbucks()
        starbucks.prep()

        # Prepare social pressure data
        soc_pres = SocialPressure()
        soc_pres.prep()

        # Prepare lenta data
        lenta = Lenta()
        lenta.prep()

        # Prepare criteo_v2 data
        criteo_v2 = CriteoV2()
        criteo_v2.prep(resample=False)
        criteo_v2.prep(resample=True)

    def prepare(self):
        """
        Call this method for preparing the data available in the data folder.
        Note: this will only prepare the data that is not protected with a password.
        """

        self.unpack_compressed_files()
        self.prepare_all_data()


if __name__ == '__main__':
    preparation_main = Preparation()
    preparation_main.prepare()
