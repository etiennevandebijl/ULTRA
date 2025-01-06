from itertools import product
from dataloader import dataloader
from sampling import get_balanced_subset, get_bm_ratio_subset

from ULTRA.ultraV2 import ultraV2

import warnings
warnings.filterwarnings('ignore')

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
        
        exp_info = {"source_dataset": source_dataset,
                    "target_dataset": target_dataset,
                    "feature_extractor": feature_extractor,
                    "version": version,
                    "protocol": protocol,
                    "uniform_sample_size": size,
                    "random_state_subset": subset_rs,
                    "experiment_name": "test ultraV2 target BM ratio 95 V2"
                    }
        
        for model_tl, model_eval,  \
            random_state_tl, random_state_eval in product(["RF"], MODELS, [0], [0, 1, 2, 3, 4]):
        
            for update_projection, update_weights in product([False, True], 
                                                             [False, True]):
            
                ultraV2(X_source, y_source, X_target, 
                        y_target = y_target, 
                        X_target_eval = X_eval, 
                        y_target_eval = y_eval, 
                        
                        experiment_info = exp_info,
                        store = True,
                       
                        #EVAL
                        model_eval = model_eval,
                        random_state_eval = 0,
                        
                        #TL
                        model_tl = model_tl,
                        random_state_tl = 0,
                        
                        #AL
                        strategy = "Random", #IF this is random, next one doesn't matter
                        model_al = "DT",
                        random_state_al = 0,
                        
                        
                        update_projection = update_projection, 
                        update_weights = update_weights,
                        
                        q = 5, 
                        T = 21,
                        uniform_tl_sample_size = 1000,
                        weighted_training_al = False,
                        weighted_training_tl = False,
                        plot_weight = False)
                
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            