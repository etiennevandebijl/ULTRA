import numpy as np

from itertools import product

from dataloader import dataloader
from sampling import get_balanced_subset
from ULTRA.ultra import ultra

import warnings
warnings.filterwarnings('ignore')

feature_extractor = "NetFlow V1"
version = "1_Raw"; 
protocol = "NF"; 
sizes = 10000

datasets = ["UNSW-NB15", "BoT-IoT","CIC-IDS-2018", "ToN-IoT"]

for source_exp, target_exp in product(datasets, datasets):
    
    X_s, y_s_mc, _, _ = dataloader(source_exp, feature_extractor, version, protocol, False, True)    
    X_d, y_d_mc, _, _ = dataloader(target_exp, feature_extractor, version, protocol, False, True)    

    for subset_rs in range(3):
        X_source, y_source = get_balanced_subset(X_s, y_s_mc, sizes, subset_rs, make_binary = True)
        X_eval, y_eval = get_balanced_subset(X_d, y_d_mc, sizes, subset_rs, make_binary= True)
        
        np.random.seed(subset_rs)
        selection = np.random.choice(np.where(y_d_mc == "Benign")[0], size = sizes)
        X_target = X_d[selection, :]
        y_target = np.zeros(sizes)

        experiment_info = {"Source Experiment": source_exp,
                            "Target Experiment": target_exp,
                            "Feature Extractor": feature_extractor,
                            "Version": version,
                            "Protocol": protocol,
                            "Sizes subsets": sizes,
                            "Random_states subsets":subset_rs,
                            "experiment-name": "active learning target benign V1"}
        print(experiment_info)
        
        for strategy in ["Random", "Certainty", "Uncertainty"]:
            for model_name in ["RF", "DT"]:
                for rs_clf in range(5):
                    for use_weights_AL in [True]:
                        ultra(X_source, y_source, X_target, 
                              model_name, 
                              rs_clf = rs_clf,
                              q = 50,
                              T = 10,
                              strategy = strategy,
                              y_target =  y_target,
                              X_target_eval = X_eval, 
                              y_target_eval = y_eval, 
                              use_weights_AL = use_weights_AL,
                              update_w = False,
                              update_A = False,
                              experiment_info = experiment_info, 
                              store = True)
