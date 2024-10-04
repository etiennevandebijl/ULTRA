import numpy as np

from dataloader import dataloader
from models import MODEL_DICT
from sampling import get_balanced_subset
from ULTRA.WIP.JDA import JDA

#%% Load data

source_exp = "UNSW-NB15"; target_exp = "CIC-IDS-2018"; feature_extractor = "NetFlow V1"
version = "1_Raw"; protocol = "NF"; model_name = "RF"

X_source, y_source_mc, _, _ = dataloader(source_exp, feature_extractor, version, protocol, False, True) 
X_target, y_target_mc, _, _ = dataloader(target_exp, feature_extractor, version, protocol, False, True)

X_source_ss, y_source_ss = get_balanced_subset(X_source, y_source_mc, 1000, 0)
X_target_ss, y_target_ss = get_balanced_subset(X_target, y_target_mc, 1000, 1)

X = np.vstack((X_source_ss, X_target_ss))
y = np.concatenate((y_source_ss, y_target_ss))

L_s = np.arange(0, 1000)
L_d = np.arange(1000, 1010)
U = np.arange(1010, 2000)

clf = MODEL_DICT["DT"]


A = JDA(X, y, L_s, L_d, U, 8, clf)

    



