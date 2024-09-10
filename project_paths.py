import os
import pandas as pd

OS_PATH = "/home/etienne/Dropbox/Projects/"
PROJECT_PATH = OS_PATH + "ULTRA/"
DATA_PATH = PROJECT_PATH + "Data/"


def get_data_folder(experiment_name, feature_extractor, version):
    """Retrieve path to set of datasets."""
    return DATA_PATH + experiment_name + "/" + feature_extractor + "/" + version + "/"


def check_existence_dataset(experiment, feature_extracter, version, protocol):
    """Check if data file exists"""
    path = get_data_folder(experiment, feature_extracter, version)
    return os.path.isfile(path + protocol + ".csv")


def get_results_folder(experiment_name, analyser, version, method):
    """Return of result folder."""
    path = PROJECT_PATH + "Results/" + experiment_name + "/" + analyser \
        + "/" + version + "/" + method + "/"
    return path


def get_results_df(experiment_name):
    return pd.read_csv(PROJECT_PATH + "Results/Tables/" + experiment_name + '.csv')


def go_or_create_folder(path, folder):
    """Path walks to path + folder and creates it is necessary."""
    if not os.path.isdir(path+folder):
        os.mkdir(path + folder)
    return path + folder + "/"