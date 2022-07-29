"""
Base IO code for all datasets
"""

import gzip
import shutil
import tarfile
from os import environ, makedirs, path
from os.path import expanduser, join
from zipfile import ZipFile

from autoum.datasets.bank_telemarketing import Bank_Telemarketing
from autoum.datasets.criteo import Criteo
from autoum.datasets.criteo_v2 import CriteoV2
from autoum.datasets.hillstrom import Hillstrom
from autoum.datasets.lenta import Lenta
from autoum.datasets.socialpressure import SocialPressure
from autoum.datasets.starbucks import Starbucks


def get_data_home(data_home: str=None) -> str:
    """Return the path of the autoum data directory.

    This folder is used by some large dataset loaders to avoid downloading the data several times.
    By default, the data directory is set to a folder named 'autoum_data' in the user home folder.

    Alternatively, it can be set by the 'AUTOUM_DATA' environment variable or programmatically by giving an explicit folder path.
    The '~' symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.

    :param data_home: The path to autoum's data directory. If `None`, the default path is `~/sklearn_learn_data`.
    :return: The path to autoum data directory
    """
    if data_home is None:
        data_home = environ.get("AUTOUM_DATA", join("~", "autoum_output"))
    data_home = expanduser(data_home)
    makedirs(data_home, exist_ok=True)
    return data_home


def unpack_zip(zip_file_path: str, dest_path: str):
    """
    Unpack a zip file

    :param zip_file_path: Path of the zip file
    :param dest_path: Path where the zip file should be unzipped
    """
    if not path.isdir(dest_path):
        makedirs(dest_path, exist_ok=True)
        with ZipFile(zip_file_path, 'r') as zip_file:
            zip_file.extractall(dest_path)


def unpack_gz(gz_file_path: str, dest_path: str, dest_file: str):
    """
    Unpack a gz file

    :param gz_file_path: Path of the gz file
    :param dest_path: Path of the directory where the gz file should be unzipped
    :param dest_file: Name of the unpacked file
    """
    if not path.exists(dest_path + dest_file):
        makedirs(dest_path, exist_ok=True)
        with gzip.open(gz_file_path, 'rb') as f_in:
            with open(dest_path + dest_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def unpack_tar(tar_file_path: str, dest_path: str):
    """
    Unpack a tar file

    :param tar_file_path: Path of the tar file
    :param dest_path: Path where the tar file should be unzipped
    """
    if not path.isdir(dest_path):
        makedirs(dest_path, exist_ok=True)
        with tarfile.open(tar_file_path) as my_tar:
            my_tar.extractall(dest_path)


def get_hillstrom_women_visit():
    """Return Hillstrom dataset with women only and visit as response variable"""
    return get_hillstrom(only_women=True, only_men=False, visit=True)


def get_hillstrom_women_conversion():
    """Return Hillstrom dataset with women only and conversion as response variable"""
    return get_hillstrom(only_women=True, only_men=False, visit=False)


def get_hillstrom_men_visit():
    """Return Hillstrom dataset with men only and visit as response variable"""
    return get_hillstrom(only_women=False, only_men=True, visit=True)


def get_hillstrom_men_conversion():
    """Return Hillstrom dataset with men only and conversion as response variable"""
    return get_hillstrom(only_women=False, only_men=True, visit=False)


def get_hillstrom_visit():
    """Return Hillstrom dataset with men and women and visit as response variable"""
    return get_hillstrom(only_women=False, only_men=False, visit=True)


def get_hillstrom_conversion():
    """Return Hillstrom dataset with men and women and conversion as response variable"""
    return get_hillstrom(only_women=False, only_men=False, visit=False)


def get_hillstrom(only_women: bool, only_men: bool, visit: bool):
    """
    Get the Hillstrom dataset

    :param only_women: True, if only the women treatment should be considered. False otherwise.
    :param only_men: True, if only the men treatment should be considered. False otherwise.
    :param visit: True, if the feature "visit" shall be used as response variable. False if the feature "conversion" shall be used as response variable.
    """
    # Unzip
    hillstrom_path_zip = "autoum/datasets/data/hillstrom.csv.zip"
    path_folder = get_data_home() + "/data/hillstrom-email/"
    unpack_zip(hillstrom_path_zip, path_folder)

    # Prepare
    hillstrom = Hillstrom(path_folder)
    data = hillstrom.prep(only_women=only_women, only_men=only_men, visit=visit)

    return data


def get_bank_telemarketing(prep_contact: bool):
    """
    Get the Bank Telemarketing dataset

    :param prep_contact: True, if the "contract" feature shall be used as repsonse. False, if the "contact" and "outcome" column shall be used as response
    """
    # Unzip
    bank_telemarketing_path_zip = "autoum/datasets/data/bank-additional-full.csv.zip"
    path_folder = get_data_home() + "/data/bank-telemarketing/"
    unpack_zip(bank_telemarketing_path_zip, path_folder)

    # Prepare
    bank_telemarketing = Bank_Telemarketing(path_folder)
    if prep_contact:
        data = bank_telemarketing.prep_contact()
    else:
        data = bank_telemarketing.prep_outcome()

    return data


def get_bank_telemarketing_contact():
    """Return Bank Telemarketing dataset with contact as response variable"""
    return get_bank_telemarketing(prep_contact=True)


def get_bank_telemarketing_outcome():
    """Return Bank Telemarketing dataset with contact and outcome as response variable"""
    return get_bank_telemarketing(prep_contact=False)


def get_criteo_v1(resample: bool):
    """
    Get the Criteo V1 dataset

    :param resample: True, if the dataset should be resampled. False otherwise
    """
    # Unzip
    criteo_path_zip = "autoum/datasets/data/criteo-uplift.csv.gz"
    path_folder = get_data_home() + "/data/criteo-v1/"
    file_name = "criteo-uplift.csv"
    unpack_gz(criteo_path_zip, path_folder, file_name)

    # Prepare
    criteo = Criteo(path_folder)
    if resample:
        return criteo.prep(resample=True)
    else:
        return criteo.prep(resample=False)


def get_criteo_v1_full():
    """Return Criteo V1 dataset"""
    return get_criteo_v1(resample=False)


def get_criteo_v1_resampled():
    """Return Criteo V1 dataset with resampling"""
    return get_criteo_v1(resample=True)


def get_criteo_v2(resample: bool):
    """
    Get the Criteo V2 dataset

    :param resample: True, if the dataset should be resampled. False otherwise
    """
    # Unzip
    criteo_path_zip = "autoum/datasets/data/criteo-uplift-v2.1.csv.gz"
    path_folder = get_data_home() + "/data/criteo-v2/"
    file_name = "criteo-uplift-v2.1.csv"
    unpack_gz(criteo_path_zip, path_folder, file_name)

    # Prepare
    criteo = CriteoV2(path_folder)
    if resample:
        return criteo.prep(resample=True)
    else:
        return criteo.prep(resample=False)


def get_criteo_v2_full():
    """Return Criteo V2 dataset"""
    return get_criteo_v2(resample=False)


def get_criteo_v2_resampled():
    """Return Criteo V2 dataset with resampling"""
    return get_criteo_v2(resample=True)


def get_lenta():
    """
    Get the Lenta dataset
    """
    # Unzip
    lenta_path_zip = "autoum/datasets/data/lenta_dataset.csv.gz"
    path_folder = get_data_home() + "/data/lenta/"
    file_name = "lenta_dataset.csv"
    unpack_gz(lenta_path_zip, path_folder, file_name)

    # Prepare
    lenta = Lenta(path_folder)
    return lenta.prep()


def get_social_pressure():
    """
    Get the Social Pressure dataset
    """
    # Unzip
    social_pressure_path_zip = "autoum/datasets/data/data.zip"
    path_folder = get_data_home() + "/data/social-pressure/"
    unpack_zip(social_pressure_path_zip, path_folder)

    # Prepare
    social_pressure = SocialPressure(path_folder)
    return social_pressure.prep()


def get_starbucks():
    """
    Get the Starbucks dataset
    """
    # Unzip
    starbucks_path_zip = "autoum/datasets/data/data.tar.gz"
    path_folder = get_data_home() + "/data/starbucks/"
    unpack_tar(starbucks_path_zip, path_folder)

    # Prepare
    starbucks = Starbucks(path_folder)
    return starbucks.prep()
