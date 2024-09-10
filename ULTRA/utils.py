import numpy as np

def check_array_is_set(array_set):
    # Check if all elements are integers
    if not all(isinstance(x, np.int64) for x in array_set):
        raise ValueError("There is a non-integer value in the set")
        
    # Check if all elements are unique
    if len(array_set) != len(set(array_set)):
        raise ValueError("There is at least one duplicate value in the set")
        
       
def normalize_weights(weights, unlabelled_indices_set):
    '''We set all unlabelledd instances at 0 and divide by sum of labelled instances'''
    check_array_is_set(unlabelled_indices_set)
    
    weights_copy = weights.copy()
    
    weights_copy[unlabelled_indices_set] = 0

    return weights_copy / np.sum(weights_copy)


def weightupdate(L_s, L_d, T, eps, loss, weights):
    ''' We update weights based on TradaBoost method'''
    
    if isinstance(L_s, list):
        L_s = np.array(L_s)
    
    if isinstance(L_d, list):
        L_d = np.array(L_d)

    if isinstance(loss, list):
        loss = np.array(loss)
    
    if isinstance(weights, list):
        weights = np.array(weights)
        
    weights_copy = weights.copy()
     
    if eps < 0 or eps > 1:
        raise ValueError("Epsilon should be between 0 and 1")
        
    check_array_is_set(L_s)
    check_array_is_set(L_d)
    
    if len(loss) != len(weights):
        raise ValueError("Loss vector length should be equal to length weights")
        
    # Determine beta for Source
    k = len(L_s)
    term = np.sqrt(2 * np.log(float(k)) / float(T))
    beta_source = 1 / (1 + term)
    
    # Determine beta for target
    beta_target = eps / (1 - eps)

    # According to the authors, we do not adjust weights of source dataset
    # if the error is higher than 0.5
    if eps > 0.5:
        beta_target = 1.0
        
    # If the error is 0, then beta_t is 0, but this breaks the update rule, 
    # so we stop the updating algorithm.
    if eps > 0.0:
        weights_copy[L_s] = weights_copy[L_s] * (beta_source) ** loss[L_s]
        weights_copy[L_d] = weights_copy[L_d] * (beta_target) ** (-1 * loss[L_d])
    
    return weights_copy



