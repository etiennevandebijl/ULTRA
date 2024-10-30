import numpy as np

from models import MODEL_DICT

from ULTRA.SSTCAplusV3 import SSTCAplusV3
from ULTRA.sampling import subset_data


def determine_loss_and_epsilon(X, y, L, L_s, L_d, A, w, p, model_tl, random_state_tl, weighted_training_tl):
    
    # ML model
    clf = MODEL_DICT[model_tl]
    
    # Set random state if possible
    if 'random_state' in clf.get_params():
        clf.set_params(random_state=random_state_tl)
    
    # Fit learner
    if weighted_training_tl:
        clf.fit(X[L] @ A, y[L], sample_weight = p[L])
    else:
        clf.fit(X[L] @ A, y[L])
            
    # Predict labelled instances (note that proba might not be optional, so we need to improve this)
    y_pred = clf.predict_proba(X[L] @ A)[:,1]

    # Default loss function
    loss = np.zeros(X.shape[0])
    
    # Determine the 
    loss[L] = np.abs(y_pred - y[L])             
    
    # I replaced the eps B with L instead of L_d
    eps_A = np.sum((w[L_d] * loss[L_d])) /  np.sum(w[L_d])
    
    return eps_A, loss

    
def optimize_projection_matrix(X, y, L, L_s, L_d, U, w, p, uniform_tl_sample_size, 
                               model_tl, random_state_tl, weighted_training_tl):
    
    # Retrieve new sets based on labelled subset
    S_T_t, L_s_subset, L_d_subset, U_subset = subset_data(L_s, L_d, U, p, uniform_tl_sample_size)
    
    B, eigenvalues, obj = SSTCAplusV3(X[S_T_t], y[S_T_t], L_s_subset, L_d_subset, U_subset, 
                                      components = 8, 
                                      k = 100, 
                                      sigma = 1.0, 
                                      lamda = 1.0, 
                                      kernel = "linear",
                                      gamma = 0.5, 
                                      mu = 1.0, 
                                      random_state = random_state_tl,
                                      semi_supervised = True, 
                                      target_dependence = True, 
                                      self_dependence = False)
        
    
    # As we selected Linear, we can simply circomvent the kernel
    B = X[S_T_t].T @ B

    eps_B, loss = determine_loss_and_epsilon(X, y, L, L_s, L_d, B, w, p, model_tl, random_state_tl, weighted_training_tl)

    print(eps_B)

    return B, eps_B, loss