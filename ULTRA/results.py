import os
import numpy as np
import pandas as pd

from project_paths import PROJECT_PATH
from evaluation import binary_evaluation_measures, BINARY_MEASURE_NAMES
from models import MODEL_DICT


def store_results(merge):
    # Store results
    storage_path = PROJECT_PATH + "Results/Tables/" + \
        merge["experiment-name"] + ".csv"
    df = pd.DataFrame.from_dict(merge, orient = "index").T    
    df.to_csv(storage_path, index=False, mode='a',
              header=not os.path.isfile(storage_path))

def evaluate_model(clf, X, y, L, L_s, L_d, U, A, experiment_info, train_info,
                   X_target_eval = None, y_target_eval = None, store = False):
    
    y_pred = clf.predict(X @ A)
    
    for test_set_name, ss in {"L": L, "L_s": L_s, "L_d": L_d, "U": U}.items():

        if len(ss) > 0:
            # TO DO:  what if U contains -1?     
            eval_metrics = binary_evaluation_measures(y[ss], y_pred[ss])
        else:
            eval_metrics = [np.nan] * 9

        dictionary = dict(zip(BINARY_MEASURE_NAMES, eval_metrics))
        test_dict = {"Test Data": test_set_name}
        merge = {**experiment_info, **train_info, ** test_dict, **dictionary}
        
        if store:
            store_results(merge)

        
    # We only determine the score on the evaluation dataset if we have their labels
    if X_target_eval is not None and y_target_eval is not None:
        
        y_pred = clf.predict(X_target_eval @ A)

        eval_metrics = binary_evaluation_measures(y_target_eval, y_pred)
    
        dictionary = dict(zip(BINARY_MEASURE_NAMES, eval_metrics))
        
        test_dict = {"Test Data": "Eval"}
        merge = {**experiment_info, **train_info, **test_dict, **dictionary}
        print(merge)
        
        if store:
            store_results(merge)

def fit_predict(X, y, L, L_s, L_d, U, A, p, model_name, experiment_info, rs_clf,
                X_target_eval = None, y_target_eval = None, store = False):
    
    # ML model
    clf = MODEL_DICT[model_name]
    
    # Set random state if possible
    if 'random_state' in clf.get_params():
        clf.set_params(random_state=rs_clf)
                
    for train_set_name, ss in {"L": L, "L_s": L_s, "L_d": L_d}.items():
        
        if train_set_name == "L_d" and len(ss) == 0:
            continue
        
        train_info = {"Train Set": train_set_name, 
                      "Weighting": False,
                      "Projection": False,
                      "Model": model_name
                      }
        
        clf.fit(X[ss], y[ss])
        evaluate_model(clf, X, y, L, L_s, L_d, U, np.eye(X.shape[1]), experiment_info, train_info,
                       X_target_eval, y_target_eval, store)

        train_info["Weighting"] = True
        print(p[ss])
        clf.fit(X[ss], y[ss], sample_weight = p[ss])                
        evaluate_model(clf, X, y, L, L_s, L_d, U, np.eye(X.shape[1]), experiment_info, train_info,
                       X_target_eval, y_target_eval, store)

        train_info["Projection"] = True

        clf.fit(X[ss] @ A, y[ss], sample_weight = p[ss])                
        evaluate_model(clf, X, y, L, L_s, L_d, U, A, experiment_info, train_info,
                       X_target_eval, y_target_eval, store)

        train_info["Weighting"] = False
        
        clf.fit(X[ss] @ A, y[ss])
        evaluate_model(clf, X, y, L, L_s, L_d, U, A, experiment_info, train_info,
                       X_target_eval, y_target_eval, store)



# =============================================================================
# Example
# =============================================================================
        
# from ML.dataloader import dataloader

# exp = "UNSW-NB15"; feature_extractor = "NetFlow V1"; version = "1_Raw"; protocol = "NF"

# X, y, _, _ = dataloader(exp, feature_extractor, version, protocol, True)    

# X = X[0:3000, :]
# y = y[0:3000]

# L_s = np.arange(0, 1000)
# L_d = np.arange(1000, 2000)
# L = np.concatenate((L_s, L_d))
# U = np.arange(2000, 3000)

# p = np.random.uniform(0, 1, 3000)

# A = np.random.uniform(size = ((X.shape[1], X.shape[1])))

# model_name = "DT"
# rs_clf = 0
# experiment_info = {"Experiment": exp,
#                 "Feature Extractor": feature_extractor,
#                 "Version": version,
#                 "Protocol": protocol,
#                 "Model": model_name}
            
# fit_predict(X, y, L,  L_s, L_d, U, A, p, model_name, experiment_info, rs_clf,
#                 X_target_eval = None, y_target_eval = None)








