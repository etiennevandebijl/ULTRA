import numpy as np

from itertools import product
from dataloader import dataloader
from sampling import get_balanced_subset, get_bm_ratio_subset
from sklearn.model_selection import train_test_split

from ULTRA.fitpredictV2 import fit_predict
from ULTRA.SSTCAplusV3 import SSTCAplusV3, project_data

import warnings
warnings.filterwarnings('ignore')

feature_extractor = "NetFlow V1"; version = "1_Raw"; protocol = "NF"; sizes = 1000

datasets = ["UNSW-NB15", "BoT-IoT", "CIC-IDS-2018", "ToN-IoT"]

bm_ratio = 0.95


def TCA_grid(X, y, L_s, L_d, U, exp_info, 
             TCA_version = None,
             random_state = None,
             semi_supervised = None):
        
    for mu, kernel in product([1, 10], ["linear", "rbf", "laplacian"]):                

        W, eigenvalues, obj = SSTCAplusV3(X, y, L_s, L_d, U,
                                          components = X.shape[0], 
                                          kernel = kernel,
                                          mu = mu, 
                                          random_state = random_state,
                                          semi_supervised = semi_supervised)
        
        for c in [2, 4, 6, 8]:
    
            
            TL_info = {"TCA Version": TCA_version,
                       "Number of components": c,
                       "Kernel": kernel,
                       "Mu": mu,
                       "random state": random_state,
                       "semi_supervised": semi_supervised,
                       "Objective score": obj,
                       "Top eigenvalue": eigenvalues[0],
                       "Sum eigenvalues": np.sum(np.abs(eigenvalues[:c])) }

            comb_info = {**exp_info, **TL_info}
            
            Z = project_data(X, X, W[:, 0:c], kernel)
            Z_eval = project_data(X_eval, X, W[:, 0:c], kernel)
            
            print(TCA_version)
            fit_predict_simpler(Z, y, L_s, L_d, U, comb_info, Z_eval, y_eval)

def fit_predict_simpler(X, y, L_s, L_d, U, dict_info, X_eval, y_eval):
    
    L = np.concatenate((L_s, L_d))
    
    A = np.eye(X.shape[1])
    p = np.ones(M_total)
    
    for rs_eval_clf, eval_model in product([0, 1, 2, 3, 4, 5], ["NN_BF", "RF", "DT", "SVM"]) :
         
        ml_info = {"Evaluation model": eval_model,
                   "Random state eval clf": rs_eval_clf}
        
        comb_info = {**dict_info, **ml_info }
        
        fit_predict(X, y, L, L_s, L_d, U, A, p, eval_model, comb_info, rs_eval_clf,
                    X_target_eval = X_eval, y_target_eval = y_eval, 
                    update_A = False, store = True)

for source_exp, target_exp in product(datasets, datasets):

    if source_exp == target_exp:
        continue
    
    X_s, y_s_mc, _, _ = dataloader(source_exp, feature_extractor, version, protocol, False, True)    
    X_d, y_d_mc, _, _ = dataloader(target_exp, feature_extractor, version, protocol, False, True)    
    
    for subset_rs in range(5):
        
        X_source, y_source = get_balanced_subset(X_s, y_s_mc, sizes, subset_rs, make_binary = True)
        X_eval, y_eval = get_balanced_subset(X_d, y_d_mc, sizes, subset_rs + 10, make_binary = True)

        X_target, y_target = get_bm_ratio_subset(X_d, y_d_mc, sizes, bm_ratio, 
                                                 subset_rs, make_binary = True)

        M_source, M_target = len(X_source), len(X_target)
        M_total = M_source + M_target
            
        X = np.vstack((X_source, X_target))
        y = np.concatenate((y_source, y_target))
    
        source_set = np.arange(0, M_source, dtype=np.int64)
        target_set = np.arange(M_source, M_total, dtype=np.int64)
    
        for labelled_size in [0, 20, 50, 100]:
        
            L_s = source_set
            
            if labelled_size == 0:
                L_d = np.empty((0), dtype=np.int64)
                U = np.arange(M_source, M_total, dtype=np.int64)
            else:
                L_d, U = train_test_split(target_set, stratify = y[target_set], train_size = labelled_size)
            
            L = np.concatenate((L_s, L_d))
                 
            exp_info = {"Source Experiment": source_exp,
                        "Target Experiment": target_exp,
                        "Feature Extractor": feature_extractor,
                        "Version": version,
                        "Protocol": protocol,
                        "Sizes subsets": sizes,
                        "Random_states subsets": subset_rs,
                        "Size L_s": len(L_s),
                        "Size L_d": len(L_d),
                        "Size U": len(U),
                        "experiment-name": "test TCA target BM ratio 95 V1"
                        }
            
            TCA_info = {"TCA version":None,
                        "Number of components": None,
                        "Kernel": None,
                        "Mu": None,
                        "random state TCA": None,
                        "semi_supervised": None, 
                        "Objective score": None,
                        "Top eigenvalue": None,
                        "Sum eigenvalues": None }
            
            
            comb_info = {**exp_info, **TCA_info}
            
            fit_predict_simpler(X, y, L_s, L_d, U, comb_info, X_eval, y_eval)

            # TCA
            TCA_grid(X, y, L_s, L_d, U, exp_info, random_state = 0,
                     TCA_version = "TCA", semi_supervised = False)
