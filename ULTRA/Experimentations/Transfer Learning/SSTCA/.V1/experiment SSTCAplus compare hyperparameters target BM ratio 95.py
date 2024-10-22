import numpy as np

from itertools import product
from dataloader import dataloader
from sampling import get_balanced_subset, get_bm_ratio_subset

from ULTRA.fitpredict import fit_predict
from ULTRA.SSTCAplus import SSTCAplus

import warnings
warnings.filterwarnings('ignore')

feature_extractor = "NetFlow V1"; version = "1_Raw"; protocol = "NF"; size = 1000

datasets = ["UNSW-NB15", "BoT-IoT", "CIC-IDS-2018", "ToN-IoT"]

bm_ratio = 0.95

for source_dataset, target_dataset in product(datasets, datasets):

    if source_dataset == target_dataset:
        continue
    
    X_s, y_s_mc, _, _ = dataloader(source_dataset, feature_extractor, version, protocol, False, True)    
    X_d, y_d_mc, _, _ = dataloader(target_dataset, feature_extractor, version, protocol, False, True)    
    
    for subset_rs in range(1):
        
        X_source, y_source = get_balanced_subset(X_s, y_s_mc, size, subset_rs, make_binary = True)
        X_eval, y_eval = get_balanced_subset(X_d, y_d_mc, size, subset_rs+10, make_binary= True)

        X_target, y_target = get_bm_ratio_subset(X_d, y_d_mc, size, bm_ratio, 
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
                
            for rs_eval_clf in range(1):
                for eval_model in ["RF"]:
                    
                    experiment_info = {"source_dataset": source_dataset,
                                "target_dataset": target_dataset,
                                "feature_extractor": feature_extractor,
                                "version": version,
                                "protocol": protocol,
                                "uniform_sample_size": size,
                                "random_state_subset": subset_rs,
                                "experiment_name": "test SSTCA target BM ratio 95",
                                "random_state_eval": rs_eval_clf,
                                "l_s_size": len(L_s),
                                "l_d_size": len(L_d),
                                "u_size": len(U),
                                }
                    
                    print(experiment_info)

                    TL_info = {"num_components": None,
                               "num_neighbours": None,
                               "sigma": None,
                               "lambda": None,
                               "gamma": None,
                               "mu": None}       
                    
                    comb_info = {**experiment_info, **TL_info }

                    fit_predict(X, y, L, L_s, L_d, U, A, p, eval_model, comb_info, rs_eval_clf,
                                    X_target_eval = X_eval, y_target_eval = y_eval, 
                                    update_A = False, store = True)

                    for k, sigma, lamda, gamma, mu in product([10, 50, 100, 250], #k 
                                                             ["MED", 1.0], # Sigma
                                                             [10, 100, 1000], # Lambda (Study)
                                                             [0.1, 0.5, 0.9], # Gamma (study) #BALANCING
                                                             [1, 10, 100] #mu (default original) 
                                                             ):

                        _, _, W = SSTCAplus(X, y, L_s, L_d, U, 
                                            X.shape[0], 
                                            k, 
                                            sigma, 
                                            lamda, "linear", gamma, mu)

                        for c in [2, 4, 8, 10]:
                            
                        
                            B = X.T @ W[:,0:c]
                                    
                        
                            TL_info = {"num_components": c,
                                       "num_neighbours": k,
                                       "sigma": sigma,
                                       "lambda": lamda,
                                       "gamma": gamma,
                                       "mu": mu}     
                            
                            comb_info = {**experiment_info, **TL_info }
                            print(comb_info)
    
                            fit_predict(X, y, L, L_s, L_d, U, B, p, eval_model, comb_info, rs_eval_clf,
                                            X_target_eval = X_eval, y_target_eval = y_eval, 
                                            update_A = True, store = True)

