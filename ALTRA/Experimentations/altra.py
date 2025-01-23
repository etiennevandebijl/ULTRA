from itertools import product

from dataloader import dataloader
from sampling import get_balanced_subset, get_bm_ratio_subset

from ALTRA.altra import altra
from ALTRA.burakfilter import burakfilter

import warnings
warnings.filterwarnings("ignore")

feature_extractor = "NetFlow V1"; version = "1_Raw"; protocol = "NF"; size = 10000

datasets = ["UNSW-NB15", "BoT-IoT","CIC-IDS-2018", "ToN-IoT"]

MODELS = ["DT", "SVM", "RF", "NN_BF"]
bm_ratio = 0.95

for source_dataset, target_dataset in product(datasets, datasets):

    if source_dataset == target_dataset:
        continue
    
    X_s, y_s_mc, _, _ = dataloader(source_dataset, feature_extractor, version, protocol, False, True)    
    X_d, y_d_mc, _, _ = dataloader(target_dataset, feature_extractor, version, protocol, False, True)    
    
    for subset_rs in range(10):
        
        X_source, y_source = get_balanced_subset(X_s, y_s_mc, size, subset_rs, make_binary = True)
        X_eval, y_eval = get_balanced_subset(X_d, y_d_mc, size, subset_rs + 10, make_binary = True)

        X_target, y_target = get_bm_ratio_subset(X_d, y_d_mc, size, bm_ratio, 
                                                 subset_rs, make_binary = True)
        
        T_ = burakfilter(X_source, X_target)
        
        exp_info = {"source_dataset": source_dataset,
                    "target_dataset": target_dataset,
                    "feature_extractor": feature_extractor,
                    "version": version,
                    "protocol": protocol,
                    "uniform_sample_size": size,
                    "random_state_subset": subset_rs,
                    "experiment_name": "test altra target BM ratio 95"
                    }
        
        for model_eval, random_state_tl, random_state_al, random_state_eval, strategy \
                in product(MODELS, [0], [0], list(range(5)), ["Random", "Uncertainty", "Certainty"]):
        
            for normalize_v, prob_predict_tl in product([False, True], [False]):
                
                                        
                altra(X_source = X_source,
                      y_source = y_source,
                      X_target = X_target,
                      y_target = y_target,
                      X_target_test = X_eval,
                      y_target_test = y_eval,
                        
                      model_eval = model_eval,
                      experiment_info = exp_info,
                      
                      T_ = T_,
                      random_state_tl = random_state_tl, 
                      random_state_al = random_state_al, 
                      random_state_eval = random_state_eval,
                      
                      sp = 0.01, 
                      sn = 5, 
                      weight_update_iterations = 5, 
                      strategy = strategy, 

                      normalize_v = normalize_v, 
                      adjust_v = True,
                      prob_predict_tl = prob_predict_tl, 
                      v_update = True, 
                      store = True)
                      
        
        



