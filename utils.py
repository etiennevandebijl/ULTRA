import datetime
import numpy as np

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


def logtransform_NetFlow_V1(X):
    for i in [2, 3, 4, 5, 6, 7]:
        X[:,i] = np.log(X[:,i] + 1)
    return X

