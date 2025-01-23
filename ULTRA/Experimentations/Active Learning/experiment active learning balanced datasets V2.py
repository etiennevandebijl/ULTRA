from itertools import product

from dataloader import dataloader
from sampling import get_balanced_subset
from ULTRA.ultraV2 import ultraV2

import warnings
warnings.filterwarnings('ignore')

#%%
feature_extractor = "NetFlow V1"; version = "1_Raw"; protocol = "NF"; size = 10000

datasets = ["UNSW-NB15", "BoT-IoT","CIC-IDS-2018", "ToN-IoT"]

MODELS = ["DT", "SVM", "RF", "NN_BF"]
STRATEGIES = ["Random", "Certainty", "Uncertainty"]

for source_dataset, target_dataset in product(datasets, datasets):
    
    X_s, y_s_mc, _, _ = dataloader(source_dataset, feature_extractor, version, protocol, False, True)    
    X_d, y_d_mc, _, _ = dataloader(target_dataset, feature_extractor, version, protocol, False, True)    

    for subset_rs in range(3):
        
        X_source, y_source = get_balanced_subset(X_s, y_s_mc, size, subset_rs, make_binary = True)
        X_target, y_target = get_balanced_subset(X_d, y_d_mc, size, subset_rs, make_binary = True)
        X_eval, y_eval = get_balanced_subset(X_d, y_d_mc, size, subset_rs + 10, make_binary = True)

        exp_info = {"source_dataset": source_dataset,
                    "target dataset": target_dataset,
                    "feature_extractor": feature_extractor,
                    "version": version,
                    "protocol": protocol,
                    "uniform_sample_size": size,
                    "random_state_subset": subset_rs,
                    "experiment_name": "test AL balanced datasets"}
        print(exp_info)
        
        for strategy, model_name, rs_clf, weighted_training_al in product(STRATEGIES, MODELS, 
                                                                    list(range(3)), [True, False]) :

        
            ultraV2(X_source, y_source, X_target, 
                  y_target = y_target, 
                  X_target_eval = X_eval, 
                  y_target_eval = y_eval, 
                  
                  experiment_info = exp_info,
                  store = True,
                 
                  #EVAL
                  model_eval = model_name,
                  random_state_eval = rs_clf,
                  
                  #TL (does not matter)
                  model_tl = model_name,
                  random_state_tl = rs_clf,
                  
                  #AL
                  strategy = strategy, 
                  model_al = model_name,
                  random_state_al = rs_clf,
                  
                  q = 100, 
                  T = 10,
                  uniform_tl_sample_size = 1000,
                  weighted_training_al = weighted_training_al,
                  weighted_training_tl = False,
                  update_projection = False,
                  update_weights = False)
