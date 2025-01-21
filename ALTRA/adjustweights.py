import numpy as np

def adjust_weights(y, T_T, v):
    '''
    This function tries to add a higher weight to positive instances
    When implementing, it turns out that this procedure does not give
    a higher weight to the positives, but to the not-majority class. 
    
    Parameters
    ----------
    y : TYPE
        List/Array with labels of instances (0 for benign, 1 for malicious).
    T_T : TYPE
        List of indices that are labelled until now.
    v : TYPE
        Weight vector for each instance.

    Returns
    -------
    v : TYPE
        Updated weight vector.

    '''
    if isinstance(y, list):
        y = np.array(y)
    
    if isinstance(T_T, list):
        T_T = np.array(T_T)
        
    if isinstance(v, list):
        v = np.array(v)
    
    if len(np.unique(T_T)) != len(T_T):
        raise ValueError("T_T should be a set of unique instances")
    
    if not all(isinstance(t_t, (int, np.int64, np.int32, np.int16, np.int8)) for t_t in T_T):
        raise ValueError("T_T should only contain integers")
    
    if np.max(T_T) >= len(y):
        raise ValueError("T_T should only contain possible y instances")
        
    if len(v) != len(T_T):
        raise ValueError("Weight vector length should be equal to labelled instances length")
    
    if np.unique(y) not in np.array([0, 1]):
        raise ValueError("y should only contain zeros and ones.")
    
    size = len(T_T)
    
    # Number of Positives/Negatives in T_T
    n_plus = np.sum(y[T_T])
    n_minus = size - n_plus
    
    # Determine weights for both type of instances
    w_plus = size / (2 * n_plus)
    w_minus = size / (2 * n_minus)

    v = [v_i * (w_plus * y[t_t] + (1 - y[t_t]) * w_minus) for v_i, t_t in zip(v, T_T) ] 

    return v




