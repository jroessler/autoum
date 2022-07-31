"""
Base IO code for all datasets
"""

import gzip
import shutil
from importlib import resources
from os import environ, makedirs, path
from os.path import expanduser, join
from zipfile import ZipFile

import requests

from autoum.datasets.criteo import Criteo
from autoum.datasets.hillstrom import Hillstrom
from autoum.datasets.lenta import Lenta
from autoum.datasets.socialpressure import SocialPressure
from autoum.datasets.starbucks import Starbucks

DATA_MODULE = "autoum.datasets.data"


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


def download_url(url, save_path, filename, chunk_size=128):
    """
    Download the given (zip) file and save it in the given save_path location

    :param url: Url of the file, which should be downloaded
    :param save_path: Path were the file should be stored
    :param filename:  Name of the file
    :param chunk_size: Chunk size
    """
    if not path.exists(save_path + filename):
        makedirs(save_path, exist_ok=True)
        r = requests.get(url, stream=True)
        with open(save_path + filename, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)


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


def unpack_zip(zip_file_name: str, dest_path: str):
    """
    Unpack a zip file

    :param zip_file_name: Path of the zip file
    :param dest_path: Path where the zip file should be unzipped
    """
    if not path.exists(dest_path):
        makedirs(dest_path, exist_ok=True)
        with resources.open_binary(DATA_MODULE, zip_file_name) as compressed_file:
            with ZipFile(compressed_file, 'r') as zip_file:
                zip_file.extractall(dest_path)


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
    hillstrom_path_zip = "hillstrom.csv.zip"
    path_folder = get_data_home() + "/data/hillstrom-email/"
    unpack_zip(hillstrom_path_zip, path_folder)

    # Prepare
    hillstrom = Hillstrom(path_folder)
    data = hillstrom.prep(only_women=only_women, only_men=only_men, visit=visit)

    return data

def get_criteo(resample: bool):
    """
    Get the Criteo dataset

    :param resample: True, if the dataset should be resampled. False otherwise
    """
    # Download zip
    path_folder = get_data_home() + "/data/criteo/"
    zip_name = "criteo-uplift-v2.1.csv.gz"
    criteo_url = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
    download_url(criteo_url, path_folder, zip_name)

    # Unzip
    file_name = "criteo-uplift-v2.1.csv"
    unpack_gz(path_folder + zip_name, path_folder, file_name)

    # Prepare
    criteo = Criteo(path_folder)
    if resample:
        return criteo.prep(resample=True)
    else:
        return criteo.prep(resample=False)


def get_criteo_full():
    """Return Criteo dataset"""
    return get_criteo(resample=False)


def get_criteo_resampled():
    """Return Criteo dataset with resampling"""
    return get_criteo(resample=True)


def get_lenta():
    """
    Get the Lenta dataset
    """

    # Download zip
    path_folder = get_data_home() + "/data/lenta/"
    zip_name = "lenta_dataset.csv.gz"
    lenta_url = "https://sklift.s3.eu-west-2.amazonaws.com/lenta_dataset.csv.gz"
    download_url(lenta_url, path_folder, zip_name)

    # Unzip
    file_name = "lenta_dataset.csv"
    unpack_gz(path_folder + zip_name, path_folder, file_name)

    # Prepare
    lenta = Lenta(path_folder)
    return lenta.prep()


def get_social_pressure():
    """
    Get the Social Pressure dataset
    """
    # Unzip
    social_pressure_path_zip = "data.zip"
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
    starbucks_zip = "starbucks.zip"
    path_folder = get_data_home() + "/data/starbucks/"
    unpack_zip(starbucks_zip, path_folder)

    # Prepare
    starbucks = Starbucks(path_folder + 'data/')
    return starbucks.prep()
