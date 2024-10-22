import numpy as np

from ULTRA.activelearning import activelearning
from sampling import get_balanced_subset

from dataloader import dataloader

exp = "UNSW-NB15"; feature_extractor = "NetFlow V1"; version = "1_Raw"; protocol = "NF"
model_name = "RF"; rs_clf = 0; M = 200; update_A = False

X, y_mc, _, _ = dataloader(exp, feature_extractor, version, protocol, False, True)    

X, y = get_balanced_subset(X, y_mc, M, 0, make_binary = True)

L_s, L_d, U = np.array_split(np.arange(M), 3)
L = np.concatenate((L_s, L_d))

A = np.eye(X.shape[1])
p = np.ones(X.shape[0])
q = 5

strategy = "Uncertainty"
activelearning(X, y, model_name, rs_clf, strategy, L, U, A, p, q, weighted_training = True)











