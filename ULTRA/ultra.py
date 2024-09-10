import numpy as np

from ULTRA.activelearning import activelearning
from ULTRA.projection import optimize_projection_matrix, determine_loss_and_epsilon

from ULTRA.utils import normalize_weights, weightupdate
from ULTRA.results import fit_predict
from ULTRA.Visualizations.weightupdate import plot_weight_update


def ultra(X_source, y_source, X_target, 
          model_name,
          rs_clf: int = 0, 
          q: int = 10, # Query size
          T: int = 50, # Number of iterations
          subset_size: int = 500, # Should Source and Target have the same?
          strategy = "Random", # Active Learning stategy
          y_target = None, # For experimentation purposes
          X_target_eval = None, # For experimentation purposes
          y_target_eval = None, # For experimentation purposes
          experiment_info = {},
          store = False,
          update_A = True, # In case we don't want to perform TL
          update_w = True, # In case we don't want to update weights
          plot_weight = False
          ):

    if X_source.shape[1] != X_target.shape[1]:
        raise ValueError("Source and Target should have the same dimensionality")
    
    input_settings = {"Model": model_name,
                      "Random state clf": rs_clf,
                      "Query size": q,
                      "Max iter": T,
                      "Subset size": subset_size,
                      "Strategy": strategy,
                      "Update A": update_A,
                      "Updaye w": update_w
                      }
    
    experiment_settings = {**experiment_info, **input_settings}
    
    # Statistics
    M_source, M_target = len(X_source), len(X_target)
    M_total = M_source + M_target

    # Stack datasets
    X = np.vstack((X_source, X_target))
    
    # Create sets of Source and Target to be able to distinguish the instances
    source_set = np.arange(0, M_source, dtype=np.int64)
    target_set = np.arange(M_source, M_total, dtype=np.int64)
    
    # In case we don't have the labels, we can create a placeholder
    if y_target is None:
        y_target = -1 * np.ones(M_target) # Default values to say its unlabelled
    
    # Create stacked y data
    y = np.concatenate((y_source, y_target))
    
    # Initialize sets
    L_s = source_set
    L_d = np.empty((0), dtype=int)
    L = np.concatenate((L_s, L_d))
    U = target_set
    
    # Initialize weight vector with only ones
    w = np.ones(M_source + M_target)

    # Default Projection matrix?
    A = np.eye(X.shape[1])
    
    w_list = [w]
    
    for t in range(T):
        
        # Determine p(t - 1, t - 1)
        p = normalize_weights(w, U)
        
        # Gather statistics
        counters = {"Size L_s": len(L_s),
                    "Size L_d": len(L_d),
                    "Size U": len(U),
                    "Iteration": t
                     }
        
        combined_info = {**experiment_settings, **counters}
        
        # We want to keep track of all information
        fit_predict(X, y, L, L_s, L_d, U, A, p, model_name, combined_info, rs_clf,
                        X_target_eval, y_target_eval, store)
        
        # Perform Active Learning
        selected = activelearning(X, y, model_name, rs_clf, strategy, L, U, A, p, q)
        
        ''' Retrieve labels in case of practical situation
        # 
        # if any(y_target == -1):
        #     # We query the selected instances to the oracle 
        #     y_target = oracle(X_target, y_target, selected)
        #     # We update the label vector
        #     y = np.concatenate((y_source, y_target))
        '''
        
        # Update L_d_t, L_t, U_t
        L_d = np.concatenate((L_d, selected))
        L = np.concatenate((L, selected))
        U = np.array([a for a in U if not a in selected])
        
        # Determine p(t - 1, t) (We have a new L here, so a new p arrived)
        p = normalize_weights(w, U)
        
        # Perform Active Learning
        if update_A:
            A, epsilon_opt, loss_opt = optimize_projection_matrix(X, y, L, L_s, 
                                                              L_d, U, w, p, subset_size, model_name, rs_clf)
        else:
            epsilon_opt, loss_opt = determine_loss_and_epsilon(X, y, L, L_s, L_d, A, w, p, model_name, rs_clf)

        if update_w:
            w = weightupdate(L_s, L_d, T, epsilon_opt, loss_opt, w)

        w_list.append(w[:])
    
        if plot_weight:
            plot_weight_update(w_list, L_s, L_d)
        

        
        
    # TO DO; build a error dependent classifier. 
    # return clf, results

 
# %% Data   

# from ML.dataloader import dataloader
# from ML.utils import select_balanced_subset

# source_exp = "UNSW-NB15"; target_exp = "BoT-IoT"; feature_extractor = "NetFlow V1"
# version = "1_Raw"; protocol = "NF"; model_name = "RF"; sizes = 10000

# X_s, y_s_mc, _, _ = dataloader(source_exp, feature_extractor, version, protocol, False, True)    
# X_d, y_d_mc, _, _ = dataloader(target_exp, feature_extractor, version, protocol, False, True)    


# X_source, y_source = select_balanced_subset(X_s, y_s_mc, sizes, 0, make_binary = True)
# X_target, y_target = select_balanced_subset(X_d, y_d_mc, sizes, 1, make_binary = True)
# X_eval, y_eval = select_balanced_subset(X_d, y_d_mc, sizes, 2, make_binary= True)

# # selection = np.random.choice(np.where(y_d_mc == "Benign")[0], size = 10000)
# # X_target = X_d[selection, :]
# # y_target = np.zeros(10000)

# experiment_info = {"Source Experiment": source_exp,
#                    "Target Experiment": target_exp,
#                    "Feature Extractor": feature_extractor,
#                    "Version": version,
#                    "Protocol": protocol,
#                    "Size Source": sizes,
#                    "Size Target": sizes,
#                    "Size eval": sizes,
#                    "Random_state Source subset": 0, 
#                    "Random_state Target subset": 1,
#                    "Random_state Eval": 2,
#                    "experiment-name": "ULTRA"}


# ultra(X_source, y_source, X_target, 
#       model_name, 
#       y_target =  y_target, 
#       T = 20,
#       X_target_eval = X_eval, y_target_eval = y_eval, strategy = "certainty",
#       experiment_info = experiment_info, store = True)


















