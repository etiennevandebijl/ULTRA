import os
import pandas as pd

from project_paths import get_data_folder, DATA_PATH, check_existence_dataset
from utils import format_ML, remove_single_instance_classes, check_data_requirements

# %% 

EXPERIMENTS = os.listdir(DATA_PATH)
FEATURE_EXTRACTORS = ["Zeek", "NetFlow V1", "NetFlow V2"]
PROTOCOLS = ["tcp", "udp", "http", "ftp", "ssh", "ssl", "dns", "NF"]

REPLACE_COLUMNS = {"IPV4_SRC_ADDR": "id.orig_h", "L4_SRC_PORT": "id.orig_p", 
                   "IPV4_DST_ADDR": "id.resp_h", "L4_DST_PORT": "id.resp_p",
                   "Attack": "Label"}

# %% read_CSV
 
def read_csv(path):
    """Read Zeek/NetFlow csv.
    
    Example: 
    data_path = get_data_folder("UNSW-NB15", "NetFlow V1", "1_Raw")
    df = read_csv(data_path + "NF.csv")
    """
    df = pd.read_csv(path)
    
    # Zeek
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
    if "ts_" in df.columns:
        df["ts_"] = pd.to_datetime(df["ts_"])
    if "duration" in df.columns:
        df["duration"] = pd.to_timedelta(df["duration"])
    
    # NetFlow
    if "Attack" in df.columns:
        df = df.drop(["Label"], axis = 1)
        df.rename(columns=REPLACE_COLUMNS, inplace=True)

    return df


# %% Data Loader

def dataloader(experiment, feature_extractor, version, protocol, binary: bool = False, 
               remove_single_instance: bool = False):
    '''

    Example:
    X, y, _, _ = dataloader("UNSW-NB15", "NetFlow V1", "1_Raw", "NF")
    '''
    if experiment not in EXPERIMENTS:
        raise ValueError("Experiment is not correct")
        
    if feature_extractor not in FEATURE_EXTRACTORS:
        raise ValueError("Feature extractor is not correct")
    
    if protocol not in PROTOCOLS:
        raise ValueError("Protocol is not correct")

    if not check_existence_dataset(experiment, feature_extractor, version, protocol):
        raise ValueError("The combination of experiment, feature extractor, version and protocol does not exist")

    data_path = get_data_folder(experiment, feature_extractor, version)
    data = read_csv(data_path + protocol + ".csv")

    # Remove classes with only 1 instance in the data    
    if remove_single_instance:
        data = remove_single_instance_classes(data)
        
    if not check_data_requirements(data):
        raise ValueError("Data is not applicable")
        
    return format_ML(data, binary)



#%% Determine the experiments for which the data exists

def retrieve_list_experiments(feature_extracter, version, protocol):
    exp_list = []
    for experiment in EXPERIMENTS:
        if check_existence_dataset(experiment, feature_extracter, version, protocol):
            exp_list.append(experiment)
    return exp_list

