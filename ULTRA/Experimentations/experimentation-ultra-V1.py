import numpy as np

from itertools import product

from ML.dataloader import dataloader
from ML.utils import select_balanced_subset
from ML.TL.ULTRA.ultra import ultra


# %%
# =============================================================================
# Experiment 1
# =============================================================================

feature_extractor = "NetFlow V1"
version = "1_Raw"; 
protocol = "NF"; 
sizes = 10000

datasets = ["UNSW-NB15", "BoT-IoT", "ToN-IoT", "CIC-IDS-2018"]

for source_exp, target_exp in product(datasets, datasets):

    if source_exp == target_exp:
        continue
    
    X_s, y_s_mc, _, _ = dataloader(source_exp, feature_extractor, version, protocol, False, True)    
    X_d, y_d_mc, _, _ = dataloader(target_exp, feature_extractor, version, protocol, False, True)    

    X_source, y_source = select_balanced_subset(X_s, y_s_mc, sizes, 0, make_binary = True)
    X_target, y_target = select_balanced_subset(X_d, y_d_mc, sizes, 1, make_binary = True)
    X_eval, y_eval = select_balanced_subset(X_d, y_d_mc, sizes, 2, make_binary= True)
    
    
    experiment_info = {"Source Experiment": source_exp,
                        "Target Experiment": target_exp,
                        "Feature Extractor": feature_extractor,
                        "Version": version,
                        "Protocol": protocol,
                        "Size Source": sizes,
                        "Size Target": sizes,
                        "Size eval": sizes,
                        "Random_state Source subset": 0, 
                        "Random_state Target subset": 1,
                        "Random_state Eval": 2,
                        "experiment-name": "ULTRA-experiment-1"}

    for strategy in ["Random", "Certainty", "Uncertainty"]:
        for model_name in ["RF"]:
            for rs_clf in range(5):
                ultra(X_source, y_source, X_target, 
                      model_name, 
                      rs_clf = rs_clf,
                      T = 30,
                      strategy = strategy,
                      y_target =  y_target,
                      X_target_eval = X_eval, 
                      y_target_eval = y_eval, 
                      experiment_info = experiment_info, store = True)

# %% 
# =============================================================================
# Experiment 3
# =============================================================================

feature_extractor = "NetFlow V1"
version = "1_Raw"; 
protocol = "NF"; 
sizes = 10000

datasets = [ "UNSW-NB15", "BoT-IoT","CIC-IDS-2018", "ToN-IoT"]

for source_exp, target_exp in product(datasets, datasets):

    if source_exp == target_exp:
        continue
    
    X_s, y_s_mc, _, _ = dataloader(source_exp, feature_extractor, version, protocol, False, True)    
    X_d, y_d_mc, _, _ = dataloader(target_exp, feature_extractor, version, protocol, False, True)    

    X_source, y_source = select_balanced_subset(X_s, y_s_mc, sizes, 0, make_binary = True)
    X_eval, y_eval = select_balanced_subset(X_d, y_d_mc, sizes, 2, make_binary= True)
    
    selection = np.random.choice(np.where(y_d_mc == "Benign")[0], size = sizes)
    X_target = X_d[selection, :]
    y_target = np.zeros(sizes)

    experiment_info = {"Source Experiment": source_exp,
                        "Target Experiment": target_exp,
                        "Feature Extractor": feature_extractor,
                        "Version": version,
                        "Protocol": protocol,
                        "Size Source": sizes,
                        "Size Target": sizes,
                        "Size eval": sizes,
                        "Random_state Source subset": 0,
                        "Random_state Target subset": 1,
                        "Random_state Eval": 2,
                        "experiment-name": "ULTRA-experiment-3"}

    for strategy in ["Random"]: # , "Random", "Certainty"
        for model_name in ["RF"]:
            for rs_clf in range(5):
                ultra(X_source, y_source, X_target, 
                      model_name, 
                      rs_clf = rs_clf,
                      q = 10,
                      T = 10,
                      strategy = strategy,
                      y_target =  y_target,
                      X_target_eval = X_eval, 
                      y_target_eval = y_eval, 
                      experiment_info = experiment_info, store = True)

