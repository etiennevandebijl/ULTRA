from dataloader import dataloader
from sampling import get_balanced_subset
from ULTRA.ultra import ultra

source_dataset = "UNSW-NB15"; target_dataset = "BoT-IoT"; feature_extractor = "NetFlow V1"
version = "1_Raw"; protocol = "NF"; model_name = "RF"; sizes = 10000

X_s, y_s_mc, _, _ = dataloader(source_dataset, feature_extractor, version, protocol, False, True)    
X_d, y_d_mc, _, _ = dataloader(target_dataset, feature_extractor, version, protocol, False, True)    


X_source, y_source = get_balanced_subset(X_s, y_s_mc, sizes, 0, make_binary = True)
X_target, y_target = get_balanced_subset(X_d, y_d_mc, sizes, 1, make_binary = True)
X_eval, y_eval = get_balanced_subset(X_d, y_d_mc, sizes, 2, make_binary= True)


experiment_info = {"source_dataset": source_dataset,
                    "target_dataset": target_dataset,
                    "feature_extractor": feature_extractor,
                    "version": version,
                    "protocol": protocol,
                    "Data Sizes": sizes,
                    "random_state_source_subset": 0, 
                    "random_state_target_subset": 1,
                    "random_state_eval": 2,
                    "experiment_name": "ULTRA"}


ultra(X_source, y_source, X_target, 
      model_name, 
      y_target = y_target, 
      T = 20,
      X_target_eval = X_eval, y_target_eval = y_eval, strategy = "Random",
      experiment_info = experiment_info, store = False)


