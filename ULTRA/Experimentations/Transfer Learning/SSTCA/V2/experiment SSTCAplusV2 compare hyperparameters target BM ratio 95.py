import numpy as np

from itertools import product
from dataloader import dataloader
from sampling import get_balanced_subset, get_bm_ratio_subset

from ULTRA.fitpredict import fit_predict
from ULTRA.SSTCAplusV2 import SSTCAplusV2

import warnings
warnings.filterwarnings('ignore')

feature_extractor = "NetFlow V1"; version = "1_Raw"; protocol = "NF"; sizes = 1000

datasets = ["UNSW-NB15", "BoT-IoT", "CIC-IDS-2018", "ToN-IoT"]

bm_ratio = 0.95

for source_exp, target_exp in product(datasets, datasets):

    if source_exp == target_exp:
        continue
    
    X_s, y_s_mc, _, _ = dataloader(source_exp, feature_extractor, version, protocol, False, True)    
    X_d, y_d_mc, _, _ = dataloader(target_exp, feature_extractor, version, protocol, False, True)    
    
    for subset_rs in range(2):
        
        X_source, y_source = get_balanced_subset(X_s, y_s_mc, sizes, subset_rs, make_binary = True)
        X_eval, y_eval = get_balanced_subset(X_d, y_d_mc, sizes, subset_rs+10, make_binary= True)

        X_target, y_target = get_bm_ratio_subset(X_d, y_d_mc, sizes, bm_ratio, 
                                                 subset_rs, make_binary = True)

        M_source, M_target = len(X_source), len(X_target)
        M_total = M_source + M_target
            
        X = np.vstack((X_source, X_target))
        y = np.concatenate((y_source, y_target))
    
        A = np.eye(X.shape[1]); p = np.ones(M_total)
        
        source_set = np.arange(0, M_source, dtype=np.int64)
        target_set = np.arange(M_source, M_total, dtype=np.int64)
    
        for labelled_size in [20, 50, 100, 200]:
        
            L_s = source_set
            L_d = np.random.choice(target_set, labelled_size, replace = False)
            U = np.setdiff1d(target_set, L_d)
            L = np.concatenate((L_s, L_d))
                
            for rs_eval_clf in range(3):
                for eval_model in ["RF"]:
                    
                    experiment_info = {"Source Experiment": source_exp,
                                        "Target Experiment": target_exp,
                                        "Feature Extractor": feature_extractor,
                                        "Version": version,
                                        "Protocol": protocol,
                                        "Sizes subsets": sizes,
                                        "Random_states subsets":subset_rs,
                                        "experiment-name": "test SSTCA-V2 target BM ratio 95",
                                        "Evaluation model": eval_model,
                                        "Random state eval clf": rs_eval_clf,
                                        "Size L_s": len(L_s),
                                        "Size L_d": len(L_d),
                                        "Size U": len(U)
                                        }
                    print(experiment_info)
                    
                    TL_info = {"Number of components": None,
                               "Neighbours": None,
                               "Sigma": None,
                               "Lambda": None,
                               "Gamma": None,
                               "Mu": None,
                               "Top eigenvalue": None,
                               "Sum eigenvalues": None}         

                    comb_info = {**experiment_info, **TL_info }

                    fit_predict(X, y, L, L_s, L_d, U, A, p, eval_model, comb_info, rs_eval_clf,
                                    X_target_eval = X_eval, y_target_eval = y_eval, 
                                    update_A = False, store = True)

                    for k, sigma, lamda, gamma, mu in product([10, 50, 100, 200], #k 
                                                             ["MED", "MEAN", 1.0], # Sigma
                                                             [0, 1, 10, 100, 1000], # Lambda (Study)
                                                             [0.5], # Gamma (study) #BALANCING
                                                             [1] #mu (default original) 
                                                             ):

                        W, eigenvalues = SSTCAplusV2(X, y, L_s, L_d, U, 
                                            X.shape[0], 
                                            k, 
                                            sigma, 
                                            lamda, "linear", gamma, mu)

                        
                        for c in [4, 8, 10]:
                            
                        
                            B = X.T @ W[:,0:c]
                                    
                            
                            TL_info = {"Number of components": c,
                                        "Neighbours": k,
                                        "Sigma": sigma,
                                        "Lambda": lamda,
                                        "Gamma": gamma,
                                        "Mu": mu,
                                        "Top eigenvalue": eigenvalues[0],
                                        "Sum eigenvalues": np.sum(eigenvalues[:c])
                                        }                                
            
                            comb_info = {**experiment_info, **TL_info }
                            print(comb_info)
    
                            fit_predict(X, y, L, L_s, L_d, U, B, p, eval_model, comb_info, rs_eval_clf,
                                            X_target_eval = X_eval, y_target_eval = y_eval, 
                                            update_A = True, store = True)





