import numpy as np

from itertools import product
from models import MODEL_DICT

from ULTRA.SSTCAplus import SSTCAplus

def subset(indices, weights, size):
    '''We select `size' indices with the highest weights '''
    
    if isinstance(indices, list):
        indices = np.array(indices)
    
    if isinstance(weights, list):
        weights = np.array(weights)
        
    if size < 1 or size > len(weights):
        raise ValueError("We cannot select more than the number of instances or 0 or less")
    
    if len(indices) != len(weights):
        raise ValueError("Weight vector length should be equal to indicices")
    
    select = np.argsort(weights)[::-1][:size]
    
    return indices[select]

def subset_data(L_s, L_d, U, p, subset_size):
    
    # Subset of all data as the projection matrix might be too large
    
    target_set = np.concatenate((L_d, U))

    S_t = subset(L_s, p[L_s], subset_size)
    T_t = subset(target_set, p[target_set], subset_size)
    S_T_t = np.concatenate((S_t, T_t))

    # This is not the best code, could be improved, but it works
    L_s_subset = np.empty((0), dtype=int)
    L_d_subset = np.empty((0), dtype=int)
    U_subset = np.empty((0), dtype=int)
    for i, s_t_t in enumerate(S_T_t):
        if s_t_t in L_s:
            L_s_subset = np.append(L_s_subset, i)
        elif s_t_t in L_d:
            L_d_subset = np.append(L_d_subset, i)
        else:
            U_subset = np.append(U_subset, i)
        
    return S_T_t, L_s_subset, L_d_subset, U_subset

def determine_loss_and_epsilon(X, y, L, L_s, L_d, A, w, p, model_name, rs_clf):
    
    # ML model
    clf = MODEL_DICT[model_name]
    
    # Set random state if possible
    if 'random_state' in clf.get_params():
        clf.set_params(random_state=rs_clf)
    
    # Fit the model
    clf.fit(X[L_s] @ A, y[L_s], sample_weight = p[L_s])

    # Predict labelled instances (note that proba might not be optional, so we need to improve this)
    y_pred = clf.predict_proba(X[L] @ A)[:,1]

    # Default loss function
    loss = np.zeros(X.shape[0])
    
    # Determine the 
    loss[L] = np.abs(y_pred - y[L])             
    
    # I replaced the eps B with L instead of L_d
    eps_A = np.sum((w[L_d] * loss[L_d])) /  np.sum(w[L_d])
    
    return eps_A, loss

    
def optimize_projection_matrix(X, y, L, L_s, L_d, U, w, p, subset_size, model_name, rs_clf):
    
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

        eps_B, loss = determine_loss_and_epsilon(X, y, L, L_s, L_d, B, w, p, model_name, rs_clf)

        print(eps_B)

        if eps_B < epsilon_opt:
            epsilon_opt = eps_B
            loss_opt = loss
            A_opt = B
            
        if eps_B == 0:
            break

        
    return A_opt, epsilon_opt, loss_opt