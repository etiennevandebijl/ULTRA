import numpy as np

from itertools import product
from models import MODEL_DICT

from ULTRA.SSTCAplus import SSTCAplus
from ULTRA.sampling import subset_data


def determine_loss_and_epsilon(X, y, L, L_s, L_d, A, w, p, model_tl, random_state_tl):
    
    # ML model
    clf = MODEL_DICT[model_tl]
    
    # Set random state if possible
    if 'random_state' in clf.get_params():
        clf.set_params(random_state=random_state_tl)
    
    # Fit the model
    clf.fit(X[L] @ A, y[L], sample_weight = p[L])

    # Predict labelled instances (note that proba might not be optional, so we need to improve this)
    y_pred = clf.predict_proba(X[L] @ A)[:,1]

    # Default loss function
    loss = np.zeros(X.shape[0])
    
    # Determine the 
    loss[L] = np.abs(y_pred - y[L])             
    
    # I replaced the eps B with L instead of L_d
    eps_A = np.sum((w[L_d] * loss[L_d])) /  np.sum(w[L_d])
    
    return eps_A, loss

    
def optimize_projection_matrix(X, y, L, L_s, L_d, U, w, p, subset_size, model_tl, random_state_tl):
    
    # Retrieve new sets based on labelled subset
    S_T_t, L_s_subset, L_d_subset, U_subset = subset_data(L_s, L_d, U, p, subset_size)
        
    # Let us search for the optimal projection matrix
    epsilon_opt = 1
    
    # We set the loss per instance at max value
    loss_opt = np.ones(X.shape[0])
    
    # We don't assess the unlabelled instances so we put them on 0
    loss_opt[U] = 0

    # Default optimal projection matrix
    A_opt = np.eye(X.shape[1])

    # We have to select some 
    for c, k, sigma, lamda, gamma, mu in product([8], 
                                             [10, 50, 100], #k 
                                             ["MED"], # Sigma
                                             [10, 100, 1000], # Lambda (Study)
                                             [0.5], # Gamma (study) #BALANCING
                                             [1.0] #mu (default original) 
                                             ):
        
        # We construct a possible projection matrix B
        _, _, B = SSTCAplus(X[S_T_t], y[S_T_t], L_s_subset, L_d_subset, U_subset, 
                            components = c, 
                            k = k, 
                            sigma = sigma,
                            lamda = lamda,
                            gamma = gamma,
                            mu = mu)
    
        # As we selected Linear, we can simply circomvent the kernel
        B = X[S_T_t].T @ B

        eps_B, loss = determine_loss_and_epsilon(X, y, L, L_s, L_d, B, w, p, model_tl, random_state_tl)

        print(eps_B)

        if eps_B < epsilon_opt:
            epsilon_opt = eps_B
            loss_opt = loss
            A_opt = B
            
        if eps_B == 0:
            break

        
    return A_opt, epsilon_opt, loss_opt