import numpy as np

from itertools import product
from dataloader import dataloader
from sampling import get_balanced_subset
from ULTRA.activelearning import activelearning
from ULTRA.fitpredict import fit_predict

import warnings
warnings.filterwarnings('ignore')

feature_extractor = "NetFlow V1"; version = "1_Raw"; protocol = "NF"; sizes = 10000

datasets = ["UNSW-NB15", "BoT-IoT","CIC-IDS-2018", "ToN-IoT"]


for source_exp, target_exp in product(datasets, datasets):

    X_s, y_s_mc, _, _ = dataloader(source_exp, feature_extractor, version, protocol, False, True)    
    X_d, y_d_mc, _, _ = dataloader(target_exp, feature_extractor, version, protocol, False, True)    
    
    for subset_rs in range(3):
        
        X_source, y_source = get_balanced_subset(X_s, y_s_mc, sizes, subset_rs, make_binary = True)
        X_target, y_target = get_balanced_subset(X_d, y_d_mc, sizes, subset_rs, make_binary= True)
        X_eval, y_eval = get_balanced_subset(X_d, y_d_mc, sizes, subset_rs+10, make_binary= True)

        M_source, M_target = len(X_source), len(X_target)
        M_total = M_source + M_target
            
        X = np.vstack((X_source, X_target))
        y = np.concatenate((y_source, y_target))
    
        A = np.eye(X.shape[1]); p = np.ones(M_total)
        
        source_set = np.arange(0, M_source, dtype=np.int64)
        target_set = np.arange(M_source, M_total, dtype=np.int64)
    
        for labelled_size in [0, 10, 100, 1000]:
        
            L_s = source_set
            L_d = np.random.choice(target_set, labelled_size, replace = False)
            U = np.setdiff1d(target_set, L_d)
            L = np.concatenate((L_s, L_d))
                
            for rs_eval_clf in range(3):
                for eval_model in ["RF", "DT"]:
                    
                    experiment_info = {"Source Experiment": source_exp,
                                        "Target Experiment": target_exp,
                                        "Feature Extractor": feature_extractor,
                                        "Version": version,
                                        "Protocol": protocol,
                                        "Sizes subsets": sizes,
                                        "Random_states subsets":subset_rs,
                                        "experiment-name": "test AL strategies",
                                        "Evaluation model": eval_model,
                                        "Random state eval clf": rs_eval_clf,
                                        "Size L_s": len(L_s),
                                        "Size L_d": len(L_d),
                                        "Size U": len(U)
                                        }
                    print(experiment_info)
                    step_info = {"strategy": None,
                                 "q": None,
                                 "al_model": None,
                                 "Random state al clf": None}    

                    comb_info = {**experiment_info, **step_info }

                    fit_predict(X, y, L, L_s, L_d, U, A, p, eval_model, comb_info, rs_eval_clf,
                                    X_target_eval = X_eval, y_target_eval = y_eval, 
                                    update_A = False, store = True)

                    for strategy in ["Random", "Uncertainty", "Certainty"]:
                        for q in [1, 5, 50]:
                            for al_model in ["RF", "DT"]:
                                for rs_al_clf in range(5):
    
                                    step_info = {"strategy": strategy,
                                                 "q": q,
                                                 "al_model": al_model,
                                                 "Random state al clf": rs_al_clf}                                
                    
                                    selected = activelearning(X, y, al_model, rs_al_clf, 
                                                              strategy, L, U, A, p, q, 
                                                              use_weight = False)
                    
                                    L_d_ = np.concatenate((L_d, selected))
                                    L_ = np.concatenate((L, selected))
                                    U_ = np.array([a for a in U if not a in selected])
                                
                                    comb_info = {**experiment_info, **step_info }

                                    fit_predict(X, y, L_, L_s, L_d_, U_, A, p, eval_model, comb_info, rs_eval_clf,
                                                    X_target_eval = X_eval, y_target_eval = y_eval, 
                                                    update_A = False, store = True)

