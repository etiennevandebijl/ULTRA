# https://github.com/formularrr/liresrc/blob/master/paper/Transfer%20Feature%20Learning%20with%20Joint%20Distribution%20Adaptation/CodeData/code/JDA.m

import numpy as np
from scipy import linalg
from sklearn.utils import check_array

from evaluation import binary_evaluation_measures

# %% 

def matrix_update(y, source_set, target_set, label):
    '''
    Example:
    y = np.array([1,1,0,1,0,1])
    source_set = np.array([0,1,2])
    target_set = np.array([3,4,5])
    matrix_update(y, source_set, target_set, 0)

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    source_set : TYPE
        DESCRIPTION.
    target_set : TYPE
        DESCRIPTION.
    label : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    ind = np.argwhere(y == label)
        
    D_source = np.intersect1d(ind, source_set)
    D_target = np.intersect1d(ind, target_set)
    
    print("Predicted label " + str(label) + " amount :" + str(len(D_target)))

    
    e = np.zeros((len(y), 1))

    e[D_source, 0] = 1 / len(D_source)
    e[D_target, 0] = -1 / len(D_target)

    M_c = e @ e.T
    return M_c

# %%
def JDA(X, y, L_s, L_d, U,
              components, 
              clf,
              lamda: float = 1., 
              random_state: int = 0):
    
    '''We assume that L_s, L_d, U indicate the instances from either source, 
    target and then labelled or unlabelled'''
    
    # Check data
    X = check_array(X)    
    
    # Set random seed
    np.random.seed(random_state)
    
    # Acquire data sizes
    M_source, M_target = len(L_s), len(L_d) + len(U)  
    M_total = M_source + M_target

    if M_total != X.shape[0]:
        raise ValueError("The sets are not equal")
    
    if components > X.shape[1]:
        raise ValueError("Components may not exceed dimensionality of problem")
        
    L = np.concatenate((L_s, L_d))
    target_set = np.concatenate((L_d, U))
    
    # Construct centering matrix H, we only have to do this once. 
    H = np.eye(M_total) - (1 / M_total) * np.ones((M_total, M_total))

    # Construct matrix M_0
    e = np.ones((M_total, 1))
    e[L_s, 0] = 1 / M_source
    e[target_set, 0] =  -1 / M_target

    M_0 = e @ e.T

    M_M = np.zeros((M_total, M_total))
    M_B = np.zeros((M_total, M_total))

    for i in range(10):

        #Grappig genoeg is dit een dimensie vs dimensie van X
        
        a = X.T @ (M_0 + M_M + M_B) @ X + lamda * np.eye(X.shape[1]) 

        b = X.T @ H @ X
        
        matrix = linalg.lstsq(a, b)[0]
        
        # Eigenvalue determination
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real
    
        sort_index = np.argsort(np.abs(eigenvalues))[::-1][:components]
        eigenvalues = eigenvalues[sort_index]
        
        A = eigenvectors[:, sort_index]
    
        Z = X @ A

        clf.fit(Z[L], y[L])
        
        y_temp = np.copy(y)        
        y_temp[U] = clf.predict(Z[U])
        
        print(binary_evaluation_measures(y[U], y_temp[U]))
        
        # Update matrix
        M_B = matrix_update(y_temp, L_s, target_set, 0)
        M_M = matrix_update(y_temp, L_s, target_set, 1)

    clf.fit(X[L], y[L])
    
    y_temp = np.copy(y)        
    y_temp[U] = clf.predict(X[U])
    
    print(binary_evaluation_measures(y[U], y_temp[U]))
    
    return A

# %%
from ML.dataloader import dataloader
from ML.TL.ULTRA.ultra import ultra
from ML.models import MODEL_DICT
from ML.utils import select_balanced_subset
import matplotlib.pyplot as plt


#%% Load data

source_exp = "UNSW-NB15"; target_exp = "UNSW-NB15"; feature_extractor = "NetFlow V1"
version = "1_Raw"; protocol = "NF"; model_name = "RF"

X_source, y_source_mc, _, _ = dataloader(source_exp, feature_extractor, version, protocol, False, True) 
X_target, y_target_mc, _, _ = dataloader(target_exp, feature_extractor, version, protocol, False, True)

X_source_ss, y_source_ss = select_balanced_subset(X_source, y_source_mc, 500, 0)
X_target_ss, y_target_ss = select_balanced_subset(X_target, y_target_mc, 500, 1)

X = np.vstack((X_source_ss, X_target_ss))
y = np.concatenate((y_source_ss, y_target_ss))

L_s = np.arange(0, 500)
L_d = np.arange(500, 750)
U = np.arange(750, 1000)

clf = MODEL_DICT["RF"]

A = JDA(X, y, L_s, L_d, U, 8, clf)

# %%


Z_source = X_source @ A
Z_target = X_target @ A

dim_1 = 0
dim_2 = 1



plt.figure(figsize = (10,10))
plt.scatter(Z_source[y_source_mc == "Benign",dim_1], Z_source[y_source_mc == "Benign",dim_2], color = "green", marker = "+")
plt.scatter(Z_source[y_source_mc != "Benign",dim_1], Z_source[y_source_mc != "Benign",dim_2], color = "red", marker = "x")
plt.scatter(Z_target[y_target_mc == "Benign",dim_1], Z_target[y_target_mc == "Benign",dim_2], color = "orange", marker = "+")
plt.scatter(Z_target[y_target_mc != "Benign",dim_1], Z_target[y_target_mc != "Benign",dim_2], color = "blue", marker = "x")

#plt.xlim(-500000, 900000)
#plt.ylim(-250, 250)

plt.show()


    
    
    
    
    
    
    
    
    
    
    
    
    
    


