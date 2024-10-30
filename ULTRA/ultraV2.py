import numpy as np

from ULTRA.activelearning import activelearning
from ULTRA.projectionV2 import optimize_projection_matrix, determine_loss_and_epsilon

from ULTRA.weightupdate import normalize_weights, weightupdate
from ULTRA.fitpredictV2 import fit_predict
from ULTRA.Visualizations.plotweightupdate import plot_weight_update


def ultraV2(X_source, y_source, X_target, 
          model_al,
          model_eval,
          model_tl,
          random_state_al: int = 0,
          random_state_eval: int = 0,
          random_state_tl: int = 0,
          q: int = 5, # Query size                                            #AL
          T: int = 20, # Number of iterations                                  #AL
          uniform_tl_sample_size: int = 100, # Should Source and Target have the same?    #TL
          strategy = "Random", # Active Learning stategy                       #AL
          y_target = None, # For experimentation purposes
          X_target_eval = None, # For experimentation purposes
          y_target_eval = None, # For experimentation purposes
          experiment_info = {},
          store = False,
          weighted_training_al = False,
          weighted_training_tl = False,
          update_projection = True, # In case we don't want to perform TL               #TL
          update_weights = True, # In case we don't want to update weights            
          plot_weight = False
          ):


    if X_source.shape[1] != X_target.shape[1]:
        raise ValueError("Source and Target should have the same dimensionality")
    
    input_settings = {"model_al": model_al,
                      "model_tl": model_tl,
                      "random_state_eval": random_state_eval,
                      "random_state_al": random_state_al,
                      "random_state_tl": random_state_tl,
                      "query_size": q,
                      "num_iterations": T,
                      "uniform_tl_sample_size": uniform_tl_sample_size,
                      "al_strategy": strategy,
                      "train_al_with_weights": weighted_training_al,
                      "train_tl_with_weights": weighted_training_tl,
                      "update_projection": update_projection,
                      "update_weights": update_weights
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
        counters = {"l_s_size": len(L_s),
                    "l_d_size": len(L_d),
                    "u_size": len(U),
                    "current_iteration": t
                     }
        print(counters)
        combined_info = {**experiment_settings, **counters}
        
        # We want to keep track of all information
        fit_predict(X, y, L, L_s, L_d, U, A, p, model_eval, combined_info, random_state_eval,
                        X_target_eval, y_target_eval, update_projection, store)
        
        # Perform Active Learning
        selected = activelearning(X, y, model_al, random_state_al, strategy, L, U, A, p, q, weighted_training_al)

        # Update L_d_t, L_t, U_t
        L_d = np.concatenate((L_d, selected))
        L = np.concatenate((L, selected))
        U = np.array([a for a in U if not a in selected])
        
        # Determine p(t - 1, t) (We have a new L here, so a new p arrived)
        p = normalize_weights(w, U)
        
        # Perform Active Learning
        if update_projection:
            A, epsilon_opt, loss_opt = optimize_projection_matrix(X, y, L, L_s, 
                                                              L_d, U, w, p, uniform_tl_sample_size, 
                                                              model_tl, random_state_tl, weighted_training_tl)
        else:
            if update_weights:
                epsilon_opt, loss_opt = determine_loss_and_epsilon(X, y, L, L_s, L_d, A, w, p, model_tl, random_state_tl, 
                                                                   weighted_training_tl)

        if update_weights:
            w = weightupdate(L_s, L_d, T, epsilon_opt, loss_opt, w)

        w_list.append(w[:])
    
        if plot_weight:
            plot_weight_update(w_list, L_s, L_d)
        

        
        
    # TO DO; build a error dependent classifier. 
    # return clf, results

 
















