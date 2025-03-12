import numpy as np

from ALTRA.burakfilter import burakfilter
from ALTRA.alsampling import ALSampling
from ALTRA.weightUpdate import weightUpdate

from ULTRA.fitpredictV2 import fit_predict

# %%

def altra(X_source, y_source, X_target, y_target, X_target_test, y_target_test, model_eval,
          experiment_info, random_state_eval, T_ = None,
          sp: float = 0.05, sn: int = 5, weight_update_iterations: int = 5, strategy = "Random", 
          random_state_tl: int = 0, random_state_al = 0, normalize_v: bool = False, adjust_v: bool = True,
          prob_predict_tl = False, v_update: bool = True, store = False):



    M_source, M_target = len(y_source), len(y_target)
    
    X = np.vstack((X_source, X_target))
    y = np.concatenate((y_source, y_target))
    
    ALNum = sp * M_target
    al_num_of_iterations = int(ALNum / sn)
    
    input_settings = {"random_state_eval": random_state_eval,
                      "random_state_al": random_state_al,
                      "random_state_tl": random_state_tl,
                      "query_size": sn,
                      "al_num_iterations": al_num_of_iterations,
                      "al_strategy": strategy,
                      "weight_update_iterations": weight_update_iterations,
                      "normalize_v": normalize_v,
                      "adjust_v": adjust_v,
                      "prob_predict_tl": prob_predict_tl,
                      "v_update": v_update
                      }
    
    experiment_settings = {**experiment_info, **input_settings}
    print(experiment_settings)
    # Initialize sets
    T_s = np.empty((0), dtype=int) #L_d
    T_u = np.arange(M_source, len(y)) # U
    # In case the Burak Filter is already applied
    if T_ is None:
        T_ = burakfilter(X_source, X_target) #L_s
    T_T = T_ # L

    v = np.ones(len(T_T))
    
    for t in range(al_num_of_iterations + 1):

        # Gather statistics
        counters = {"l_s_size": len(T_),
                    "l_d_size": len(T_s),
                    "u_size": len(T_u),
                    "current_iteration": t
                     }
        combined_info = {**experiment_settings, **counters}

        # check p, model_eval, xperiment_info, random_state_eval,
        p = np.zeros(len(y))
        p[T_T] = v
        
        fit_predict(X, y, T_T, T_, T_s, T_u, None, p, 
                    model_eval, combined_info, random_state_eval,
                    X_target_eval = X_target_test, y_target_eval = y_target_test, 
                    update_projection = False, store = store)

        selected = ALSampling(X, y, T_T, T_u, v, sn, strategy, random_state_al, adjust_v)

        # Update the sets
        T_s = np.concatenate((T_s, selected))
        T_u = [a for a in T_u if not a in selected]
        T_T = np.concatenate((T_T, selected))

        v = np.ones(len(T_T))  
        
        if v_update:
            v_update_list, eps_list = weightUpdate(X, y, T_T, T_s, v,
                                                   weight_update_iterations, random_state_tl, 
                                                   adjust_v, normalize_v, prob_predict_tl)
            v = v_update_list[-1]



