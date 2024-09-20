import numpy as np

from sklearn.model_selection import train_test_split


def get_random_subsample_indices(y, subsample_size, random_state):
    '''Sample randomly sample_size instances from y_train
    
    Example:
    y = np.random.choice([0, 1], size=(100))
    subsample_size = 10
    random_state = 0
    get_random_subsample_indices(y, subsample_size, random_state)
    '''
    np.random.seed(random_state)
    
    benign_sample = np.random.choice(np.flatnonzero(y == 0))
    malicious_sample = np.random.choice(np.flatnonzero(y == 1))
    
    # At least select one benign and one malicious
    initial_samples = np.array([benign_sample, malicious_sample])
    
    # Determine other options
    remaining_indices = np.delete(np.arange(len(y)), initial_samples)
    
    additional_samples = np.random.choice(remaining_indices, size = subsample_size - 2, replace = False)
    final_samples = np.sort(np.concatenate((initial_samples, additional_samples)))
    
    return final_samples


def equalize_by_removing_majority(X, y_mc, random_state = 0):
    '''
    Function to balance malicious and benign in a stratefied manner.
    
    Example:
    
    X = np.random.uniform(size = (100, 2))
    y_mc = np.random.choice(["Benign", "Malicious_1", "Malicious_2"], size = 100)
    X, y_mc = equalize_by_removing_majority(X, y_mc, random_state = 0)
    '''
    benign_instances = np.flatnonzero(y_mc == "Benign")
    malicious_instances = np.flatnonzero(y_mc != "Benign")
    
    if len(malicious_instances) > len(benign_instances):
        malicious_instances,_ = train_test_split(malicious_instances,
                                                 stratify = y_mc[malicious_instances],
                                                 train_size = len(benign_instances),
                                                 random_state=random_state)

    else:
        np.random.seed(random_state)
        benign_instances = np.random.choice(benign_instances, 
                                      size = len(malicious_instances),
                                      replace = False)

    selected = np.sort(np.concatenate((benign_instances, malicious_instances)))
    
    return X[selected], y_mc[selected]


def get_balanced_subset(X, y_mc, size, random_state, make_binary: bool = True):
    
    X, y_mc = equalize_by_removing_majority(X, y_mc)
    X, _, y_mc, _ = train_test_split(X, y_mc, stratify = y_mc,
                                     train_size = size, random_state=random_state)

    if make_binary:
        y_mc = np.where(y_mc == "Benign", 0, 1) 
    return X, y_mc

def select_benign_subset(X, y_mc, size, random_state, make_binary: bool = True):
    '''Select `size' instances of benign instances
    
    Example:
    X = np.random.uniform(size = (100, 2))
    y_mc = np.random.choice(["Benign", "Malicious_1", "Malicious_2"], size = 100)
    size = 20; random_state = 0
    X, y_mc = select_benign_subset(X, y_mc, size, random_state)
    '''
    benign_instances = np.flatnonzero(y_mc == "Benign")
    
    if size > len(benign_instances):
        raise ValueError("We cannot select more instances")
    
    np.random.seed(random_state)
    selected = np.random.choice(benign_instances, 
                                size = size,
                                replace = False)
    
    if make_binary:
        y_mc = np.where(y_mc == "Benign", 0, 1) 
        
    return X[selected], y_mc[selected]


def select_malicious_subset(X, y_mc, size, random_state, make_binary: bool = True):
    '''Select `size' instances of malicious instances
    
    Example:
    X = np.random.uniform(size = (100, 2))
    y_mc = np.random.choice(["Benign", "Malicious_1", "Malicious_2"], size = 100)
    size = 20; random_state = 0
    X, y_mc = select_malicious_subset(X, y_mc, size, random_state)
    '''
    malicious_instances = np.flatnonzero(y_mc != "Benign")
    
    if size > len(malicious_instances):
        raise ValueError("We cannot select more instances")
    
    selected,_ = train_test_split(malicious_instances,
                                             stratify = y_mc[malicious_instances],
                                             train_size = size,
                                             random_state=random_state)
    if make_binary: 
        y_mc = np.where(y_mc == "Benign", 0, 1) 
    return X[selected], y_mc[selected]
