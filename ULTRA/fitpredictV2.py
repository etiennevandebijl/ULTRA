import os
import numpy as np
import pandas as pd

from project_paths import PROJECT_PATH
from evaluation import binary_evaluation_measures, BINARY_MEASURE_NAMES
from models import MODEL_DICT

from inspect import signature

from sklearn.metrics import roc_auc_score


def store_results(experiment_dict):
    '''Store results'''

    storage_path = PROJECT_PATH + "Results/Tables/" + \
        experiment_dict["experiment_name"] + ".csv"
    df = pd.DataFrame.from_dict(experiment_dict, orient = "index").T    
    df.to_csv(storage_path, index=False, mode='a',
              header=not os.path.isfile(storage_path))

def evaluate_model(clf, X, y, L, L_s, L_d, U, A, experiment_info, train_info,
                   X_target_eval = None, y_target_eval = None, store = False):
    
    y_pred = clf.predict(X @ A)
    
    if hasattr(clf, 'predict_proba'):
        y_pred_proba = clf.predict_proba(X @ A)[:,1]    
    
    for test_set_name, ss in {"L": L, "L_s": L_s, "L_d": L_d, "U": U}.items():

        eval_metrics = [np.nan] * len(BINARY_MEASURE_NAMES)
        if len(ss) > 0:   
            eval_metrics = binary_evaluation_measures(y[ss], y_pred[ss])
        results = dict(zip(BINARY_MEASURE_NAMES, eval_metrics))

        if hasattr(clf, 'predict_proba') and len(ss) > 0 and len(np.unique(y[ss])) == 2:
            results["roc_auc"] = roc_auc_score(y[ss], y_pred_proba[ss])
        else:
            results["roc_auc"] = np.nan
        
        test_dict = {"test_set": test_set_name}
        experiment_dict = {**experiment_info, **train_info, **test_dict, **results}

        if store:
            store_results(experiment_dict)

        
    # We only determine the score on the evaluation dataset if we have their labels
    if X_target_eval is not None and y_target_eval is not None:
        
        y_pred = clf.predict(X_target_eval @ A)

        eval_metrics = binary_evaluation_measures(y_target_eval, y_pred)
        results = dict(zip(BINARY_MEASURE_NAMES, eval_metrics))
        
        if hasattr(clf, 'predict_proba'):
            results["roc_auc"] = roc_auc_score(y_target_eval, clf.predict_proba(X_target_eval @ A)[:,1])
        else:
            results["roc_auc"] = np.nan
        
        test_dict = {"test_set": "Eval"}
        experiment_dict = {**experiment_info, **train_info, **test_dict, **results}

        if store:
            store_results(experiment_dict)


def fit_predict(X, y, L, L_s, L_d, U, A, p, model_eval, experiment_info, random_state_eval,
                X_target_eval = None, y_target_eval = None, 
                update_projection = True, store = False):

    # ML model
    clf = MODEL_DICT[model_eval]
    
    # Set random state if possible
    if 'random_state' in clf.get_params():
        clf.set_params(random_state=random_state_eval)
                
    for train_set_name, ss in {"L": L, "L_s": L_s, "L_d": L_d}.items():
        
        if train_set_name == "L_d" and len(ss) == 0:
            continue
        
        if not len(np.unique(y[ss])) > 1:
            continue
        
        train_info = {"training_set": train_set_name, "model_eval": model_eval,
                      "train_eval_with_weights": False,
                      "train_eval_with_projection": False
                      }
        
        clf.fit(X[ss], y[ss])
        evaluate_model(clf, X, y, L, L_s, L_d, U, np.eye(X.shape[1]), experiment_info, train_info,
                       X_target_eval, y_target_eval, store)

        train_info["train_eval_with_weights"] = True

        if "sample_weight" in signature(clf.fit).parameters:
            clf.fit(X[ss], y[ss], sample_weight = p[ss])                
            evaluate_model(clf, X, y, L, L_s, L_d, U, np.eye(X.shape[1]), experiment_info, train_info,
                           X_target_eval, y_target_eval, store)

        if update_projection:
            
            train_info["train_eval_with_projection"] = True
    
            if "sample_weight" in signature(clf.fit).parameters:
                clf.fit(X[ss] @ A, y[ss], sample_weight = p[ss])                
                evaluate_model(clf, X, y, L, L_s, L_d, U, A, experiment_info, train_info,
                               X_target_eval, y_target_eval, store)
        
            train_info["train_eval_with_weights"] = False
            
            clf.fit(X[ss] @ A, y[ss])
            evaluate_model(clf, X, y, L, L_s, L_d, U, A, experiment_info, train_info,
                           X_target_eval, y_target_eval, store)









