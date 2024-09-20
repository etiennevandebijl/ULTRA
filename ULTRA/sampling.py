import numpy as np

def subset_highest_weights(indices, weights, size):
    ''' Select `size' instances with the highest weights
    
    Example:
        
    indices = np.arange(100)
    weights = np.random.uniform(0, 1, size = 100)
    size = 10
    subset_highest_weights(indices, weights, size)
    
    Return:
        corresponding indices
    '''
    
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


def subset_data(L_s, L_d, U, p, size):
    ''' Select `size' instances from both Source and Target dataset on highest weights 
    
    Example:
    p = np.random.uniform(0, 1, 100)
    L_s = np.arange(0, 50)
    L_d = np.arange(50, 60)
    U = np.arange(60, 100)
    size = 25
    subset_data(L_s, L_d, U, p, size)
    
    Return: selection indicies and updated sets

    '''

    # Combined sets labelled and unlabelled target data    
    target_set = np.concatenate((L_d, U))

    # Select size instances
    S_t = subset_highest_weights(L_s, p[L_s], size)
    T_t = subset_highest_weights(target_set, p[target_set], size)
    
    # Combine sets
    data_subset = np.concatenate((S_t, T_t))

    # This is not the best code, could be improved, but it works
    L_s_subset = np.empty((0), dtype=int)
    L_d_subset = np.empty((0), dtype=int)
    U_subset = np.empty((0), dtype=int)
    
    for i, s_t_t in enumerate(data_subset):
        if s_t_t in L_s:
            L_s_subset = np.append(L_s_subset, i)
        elif s_t_t in L_d:
            L_d_subset = np.append(L_d_subset, i)
        else:
            U_subset = np.append(U_subset, i)


    return data_subset, L_s_subset, L_d_subset, U_subset
