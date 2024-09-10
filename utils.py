import datetime
import numpy as np

from sklearn.model_selection import train_test_split


IGNORE_COLS = ["uid", "ts", "ts_", "id.orig_h", "id.resp_h", "id.resp_p",
               "id.orig_p", "local_orig", "local_resp"]


def format_ML(df, binary=False):
    """Convert df to ML ready format."""
    df = df.drop(columns = [c for c in IGNORE_COLS if c in df.columns], axis = 1)

    if "duration" in df.columns:
        df["duration"] = df["duration"] / datetime.timedelta(seconds=1)

    if binary:
        df["Label"] = np.where(df["Label"] != "Benign", 1, 0)

    X = df.drop(columns = "Label", axis = 1)
    feature_names = X.columns.tolist()

    X = X.values * 1.0
    # Update from 4 march 2024 as there is infinity that we cannot handle
    X = np.where(np.isinf(X.astype(np.float32)), np.finfo(np.float32).max, X)

    y = np.ravel(df["Label"])
    labels = np.unique(y)
    if binary:
        labels = [0, 1]
    return X, y, feature_names, labels


def remove_single_instance_classes(data):
    ''' Remove classes where only 1 instance occurs'''
    one_instance_classes = data["Label"].drop_duplicates(keep=False).tolist() 
    data = data[~data["Label"].isin(one_instance_classes)] #need at least two instances for each class
    return data


def check_data_requirements(data):
    '''Check if data consists of more than 100 instances and at least 2 classes'''
    return (data.shape[0] > 1000 and len(data["Label"].unique()) > 1)


def random_subsample_indices(random_state, y_train, sample_size):
    '''Sample randomly sample_size instances from y_train'''
    np.random.seed(random_state)
    
    benign_inst = np.random.choice([i for i, label in enumerate(y_train) if label == 0])
    malicious_inst = np.random.choice([i for i, label in enumerate(y_train) if label == 1])
    
    options_list = list(range(len(y_train)))
    del options_list[benign_inst - 1]
    del options_list[malicious_inst - 1]
    
    options = list(np.random.choice(options_list, size = sample_size - 2, replace = False))
    options = options + [benign_inst, malicious_inst]
    
    return options


def equalize_by_removing_majority(X, y_mc):
    '''
    Function to balance malicious and benign in a stratefied manner.

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    y_mc : TYPE
        DESCRIPTION.

    Returns
    -------
    X : TYPE
        DESCRIPTION.
    y_mc : TYPE
        DESCRIPTION.

    '''
    benign_instances = np.where(y_mc == "Benign")
    malicious_instance = np.where(y_mc != "Benign")
    
    if len(malicious_instance[0]) / len(y_mc) > 0.5:
        
        X_malicious, _, y_malicious, _ = train_test_split(X[malicious_instance],
                                                    y_mc[malicious_instance],
                                                    stratify = y_mc[malicious_instance],
                                                    train_size = len(benign_instances[0]),
                                                    random_state=0)
        
        X = np.concatenate((X[benign_instances], X_malicious), axis=0)
        y_mc = np.concatenate((y_mc[benign_instances], y_malicious), axis=0)
    else:
        X_benign, _, y_benign, _ = train_test_split(X[benign_instances],
                                                    y_mc[benign_instances],
                                                    train_size = len(malicious_instance[0]),
                                                    random_state=0)
        
        X = np.concatenate((X[malicious_instance], X_benign), axis=0)
        y_mc = np.concatenate((y_mc[malicious_instance], y_benign), axis=0)
        
    return X, y_mc

def select_balanced_subset(X, y_mc, size, random_state, make_binary: bool = True):
    
    X, y_mc = equalize_by_removing_majority(X, y_mc)
    X, _, y_mc, _ = train_test_split(X, y_mc, stratify = y_mc,
                                                   train_size = size, random_state=random_state)
    
    if make_binary:
        y_mc = np.where(y_mc == "Benign", 0, 1) 
    return X, y_mc

def select_benign_subset(X, y_mc, size, random_state, make_binary: bool = True):
    benign_instances = np.where(y_mc == "Benign")
    
    X, _, y_mc, _ = train_test_split(X[benign_instances],
                                     y_mc[benign_instances],
                                     train_size = size,
                                     random_state=random_state)
    if make_binary:
        y_mc = np.where(y_mc == "Benign", 0, 1) 
    return X, y_mc


def select_malicious_subset(X, y_mc, size, random_state, make_binary: bool = True):
    malicious_instances = np.where(y_mc != "Benign")
    
    X, _, y_mc, _ = train_test_split(X[malicious_instances],
                                     y_mc[malicious_instances],
                                     stratify = y_mc[malicious_instances],
                                     train_size = size,
                                     random_state=random_state)
    if make_binary:
        y_mc = np.where(y_mc == "Benign", 0, 1) 
    return X, y_mc


def logtransform_NetFlow_V1(X):
    for i in [2, 3, 4, 5, 6, 7]:
        X[:,i] = np.log(X[:,i] + 1)
    return X

