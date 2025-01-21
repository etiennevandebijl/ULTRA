import numpy as np
from tqdm import tqdm
from sklearn.utils import check_array

def burakfilter(X_source, X_target, top_k:int = 10):
    '''
    This function simply selects the top k closests instances (in the source 
    dataset) for each instance in the target dataset. A nearest neighbour kind
    of approach. This is implemented as a deterministic algortihm, so no random state
    is required. 

    Parameters
    ----------
    X_source : TYPE
        DESCRIPTION.
    X_target : TYPE
        DESCRIPTION.
    top_k : int, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    T_ : Subset of indices closest instances
        DESCRIPTION.

    '''
    X_source = check_array(X_source)
    X_target = check_array(X_target)
    
    if top_k < 1:
        raise ValueError("top_k should at least be 1")
    
    T_ = []
    for x_i_target in tqdm(X_target, total = X_target.shape[0]):
        distances = [np.linalg.norm(x_j_source - x_i_target) for x_j_source in X_source]
        selected = np.argsort(distances)[:top_k]
        T_.append(selected)
    T_ = np.unique(T_)
    
    return T_

