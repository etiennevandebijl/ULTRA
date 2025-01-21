import numpy as np

from sklearn import svm
from sklearn.utils import check_array
# from ML.evaluation import print_binary_results
from ALTRA.adjustweights import adjust_weights

#%% 

def ALSampling(X, y, T_T, T_u, v, sn: int = 5, strategy = "Random", 
               random_state_al: int = 0, adjust_v: bool = True):
    '''
    Query instances from the set of unlabelled instances. We implemented three 
    versions, two of which are described by the authors of ALTRA. That is,
    Certainty and Uncertainty. The default is random, but otherwise it is either
    of those two mentioned methods.

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    T_T : TYPE
        List with index of labelled instances (source and target).
    T_u : TYPE
        List with index of unlabelled target instances.
    v : TYPE
        Weight vector of each labelled instance.
    sn : int, optional
        Sampling size. The default is 5.
    strategy : TYPE, optional
        Given our SVM model, do we sample based on certainty, furtherest from 
        hyperplane, uncertatiny, clostest, or random. The default is "Certainty".
    random_state_svm : int, optional
        DESCRIPTION. The default is 0.
    adjust_v : bool, optional
        Whether or not to adjust v by dominance. The defauly is True. 

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    # Check format X and y
    X = check_array(X)

    if isinstance(y, list):
        y = np.array(y)
    
    if np.unique(y) not in np.array([0, 1]):
        raise ValueError("y should only contain zeros and ones.")

    if X.shape[0] != len(y):
        raise ValueError("Number of instances should be equal (X,y)")

    if isinstance(T_T, list):
        T_T = np.array(T_T)

    if len(np.unique(T_T)) != len(T_T):
        raise ValueError("T_T should be a set of unique instances")
    
    if not all(isinstance(t_t, (int, np.int64, np.int32, np.int16, np.int8)) for t_t in T_T):
        raise ValueError("T_T should only contain integers")

    if isinstance(T_u, list):
        T_u = np.array(T_u)
        
    if len(np.unique(T_u)) != len(T_u):
        raise ValueError("T_u should be a set of unique instances")
    
    if not all(isinstance(t_u, (int, np.int64, np.int32, np.int16, np.int8)) for t_u in T_u):
        raise ValueError("T_u should only contain integers")

    if len(T_T) + len(T_u) > len(y):
        raise ValueError("Length labelled + unlabelled should not exceed y")

    if len(np.intersect1d(T_u, T_T)) > 0:
        raise ValueError("There shouldn't be an overlap between T_T and T_u")
    
    if isinstance(v, list):
        v = np.array(v)
    
    if len(v) != len(T_T):
        raise ValueError("v and T_T should have equal length")

    if sn < 1 or sn > len(T_u) :
        raise ValueError("sn should at least be 1 and smaller than len(T_u)")        

    if random_state_al < 0:
        raise ValueError("Random state svm should be positive")

    if not strategy in ["Random", "Certainty", "Uncertainty"]:
        raise ValueError("This strategy is not implemented")

    if strategy == "Random":
        np.random.seed(0)
        return np.random.choice(T_u, size = sn, replace = False)
    
    # Retrieve Weighted SVM model with hinge loss
    clf = svm.LinearSVC(loss="hinge", random_state = random_state_al)
    
    # Fit model on weighted labelled instances
    if adjust_v:
        clf.fit(X[T_T], y[T_T], adjust_weights(y, T_T, v))
    else:
        clf.fit(X[T_T], y[T_T], v)

    # Look at the results of our model.
    # print_binary_results(y[T_u], clf.predict(X[T_u]))

    # Look at the distance of instances to the decision function
    y_pred = clf.decision_function(X[T_u])

    # Sort instances on distance (could be positive or negative, so we
    # take the absolute)
    selected = T_u[np.argsort(np.abs(y_pred))]

    # Sort on certainty or uncertainty
    if strategy == "Uncertainty":
        selected = selected[::-1]

    return selected[:sn]


