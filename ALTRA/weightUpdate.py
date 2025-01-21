import numpy as np

from sklearn import svm
from sklearn.utils import check_array

from ALTRA.adjustweights import adjust_weights

def weightUpdate(X, y, T_T, T_s, v, weight_update_iterations: int = 5, random_state_tl: int = 0,
                 adjust_v: bool = True, normalize_v: bool = False,
                 prob_predict_tl = False):
    '''
    Weight update of vector v using a variant of TrAdaBoost

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    T_T : TYPE
        This is a list of labelled instances (both source and target).
    T_s : TYPE
        The is a list of labelled instances of target dataset.
    v : TYPE
        Weight vector.
    iter_ : TYPE
        Maximum number of iterations.

    Returns
    -------
    v : TYPE
        New weight vector.
    eps_list : TYPE
        List of error.

    '''
    
    X = check_array(X)
    
    if isinstance(y, list):
        y = np.array(y)
    
    if np.unique(y) not in np.array([0, 1]):
        raise ValueError("y should only contain zeros and ones.")
    
    if isinstance(T_T, list):
        T_T = np.array(T_T)
    
    if len(np.unique(T_T)) != len(T_T):
        raise ValueError("T_T should be a set of unique instances")
    
    if not all(isinstance(t_t, (int, np.int64, np.int32, np.int16, np.int8)) for t_t in T_T):
        raise ValueError("T_T should only contain integers")
    
    if isinstance(T_s, list):
        T_s = np.array(T_s)

    if len(np.unique(T_s)) != len(T_s):
        raise ValueError("T_s should be a set of unique instances")
    
    if not all(isinstance(t_s, (int, np.int64, np.int32, np.int16, np.int8)) for t_s in T_s):
        raise ValueError("T_s should only contain integers")
    
    if len(T_s) < 1:
        raise ValueError("There should at least be one instance labelled in target")
    
    if len(np.setdiff1d(T_s, T_T)) > 0:
        raise ValueError("T_s should be in T_T")
    
    if isinstance(v, list):
        v = np.array(v)
    
    if len(v) != len(T_T):
        raise ValueError("v and T_T should have equal length")
        
    if weight_update_iterations < 0:
        raise ValueError("Number of iterations must be positive")
    
    if random_state_tl< 0:
        raise ValueError("Random state SVM must be positive")
    
    eps_list = []
    
    if normalize_v:
        v_sum = np.sum(v)
        v = [a / v_sum for a in v]

    v_list = [v]

    for _ in range(weight_update_iterations):
        
        # Retrieve weighted SVM model
        if prob_predict_tl:
            clf = svm.SVC(kernel='linear', probability=True, random_state = random_state_tl)
        else:
            clf = svm.LinearSVC(loss="hinge", random_state = random_state_tl)
        
        # Fit in labelled data with the given weight vector
        if adjust_v:
            clf.fit(X[T_T], y[T_T], sample_weight = adjust_weights(y, T_T, v))
        else:
            clf.fit(X[T_T], y[T_T], sample_weight = v)

        # We predict binary, but could have also used confidence bounds if possible
        if prob_predict_tl:
            y_pred = clf.predict_proba(X[T_T])
        else:
            y_pred = clf.predict(X[T_T])
        
        
        errors = [np.abs(y_pred[i] - y[t_t]) for i, t_t in enumerate(T_T)]
        
        # Determine the error on the labelled target data
        sum_weights_T_s_up = np.sum([v_i * e for v_i, t_t, e in zip(v, T_T, errors) if t_t in T_s])
        sum_weights_T_s = np.sum([v_i for v_i, t_t in zip(v, T_T) if t_t in T_s]) 
        eps_t = sum_weights_T_s_up /  sum_weights_T_s
        eps_list.append(eps_t)
        
        # print(eps_t)
       
        # Step 6: determine beta's 
        k = len(T_T) - len(T_s)
        term = np.sqrt(2 * np.log(k) / weight_update_iterations)
        beta = 1 / (1 + term)
        beta_t = eps_t / (1 - eps_t)

        # According to the authors, we do not adjust weights of source dataset
        # if the error is higher than 0.5
        if eps_t > 0.5:
            beta_t = 1.0
            
        # If the error is 0, then beta_t is 0, but this breaks the update rule, 
        # so we stop the updating algorithm.
        if eps_t == 0:
            break

        v = [v_i * pow(beta_t, -1 * e)  if t_t in T_s else v_i * pow(beta, e) for v_i, t_t, e in zip(v, T_T, errors)]

        # Optional, but not used by authors:
        if normalize_v:
            v_sum = np.sum(v)
            v = [a / v_sum for a in v]

        v_list.append(v[:])
    
    return v_list, eps_list







