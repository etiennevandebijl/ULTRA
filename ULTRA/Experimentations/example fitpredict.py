import numpy as np

from ULTRA.fitpredict import fit_predict

from dataloader import dataloader

exp = "UNSW-NB15"; feature_extractor = "NetFlow V1"; version = "1_Raw"; protocol = "NF"
model_name = "RF"; rs_clf = 0; M = 1000; update_A = False

X, y, _, _ = dataloader(exp, feature_extractor, version, protocol, True)    

X = X[0:M, :]
y = y[0:M]

L_s, L_d, U = np.array_split(np.arange(M), 3)
L = np.concatenate((L_s, L_d))

# Weighting and Active Learning
A = np.eye(X.shape[1])
p = np.ones(M) / M 

experiment_info = {"Experiment": exp,
                "Feature Extractor": feature_extractor,
                "Version": version,
                "Protocol": protocol,
                "Model": model_name,
                "experiment-name":"Example",
                "Update A": update_A}
            
fit_predict(X, y, L,  L_s, L_d, U, A, p, model_name, experiment_info, rs_clf,
                X_target_eval = None, y_target_eval = None, update_A = update_A,
                store= True)


