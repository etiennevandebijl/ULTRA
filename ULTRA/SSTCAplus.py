import numpy as np

from sklearn.utils import check_array
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import euclidean_distances, pairwise_kernels

from scipy import linalg

def SSTCAplus(X, y, L_s, L_d, U,
              components, 
              k, 
              sigma, 
              lamda, 
              kernel = "linear",
              gamma: float = 0.5, 
              mu: float = 1., 
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
        
    # Construct matrix K 
    K = pairwise_kernels(X, X, metric = kernel) # Other option

    # Construct matrix L
    e = np.ones((M_total, 1))
    e[L_s, 0] = 1 / M_source
    e[L_d, 0] =  -1 / M_target
    e[U, 0] = -1 / M_target  

    L = e * e.T

    # Construct centering matrix H
    H = np.eye(M_total) - (1 / M_total) * np.ones((M_total, M_total))

    # maximize the dependence between the extracted features and the class labels

    Kyy = (y.reshape(-1, 1) == y).astype(int)    
    
    #Exclude non-labelled
    a = np.ones((M_total, 1))
    a[U, 0] = 0
    #a[L_d, 0] = 0
    O = a * a.T
    
    # Moet ik de diagonal niet op 1 zetten?    
    O = np.maximum(O,np.eye(M_total))
    
    Kyy = np.multiply(Kyy,O)

    Kyy_star = gamma * Kyy + (1 - gamma) * np.eye(M_total)
    
    # locality preservation
    # https://en.wikipedia.org/wiki/Euclidean_distance
    dist_matrix = euclidean_distances(X)
    
    knn_graph = kneighbors_graph(X, n_neighbors = k, 
                                 mode='connectivity', include_self=False).toarray()
    
    # Make the graph symmetric
    knn_graph = np.maximum(knn_graph, knn_graph.T)
    
    # Semisupervised TCA for DA in Remote Sensing
    if sigma == "MED":
        sigma = np.median(X)

    # Affinity/adjecency  matrix (RBF ?)
    M = np.exp(- dist_matrix**2 / (2 * sigma**2)) * knn_graph
    
    # Degree matrix ()
    D = np.sum(M, axis = 0) * np.eye(M_total)
    
    Laplace = D - M

    # This is more computationally stable
    a = K @ (L + lamda * Laplace) @ K + mu * np.eye(M_total)
    b = K @ H @ Kyy_star @ H @ K
    
    matrix = linalg.lstsq(a, b)[0]

    # Retrieve eigenvalues and eigenvectors
    # Important assumption, as we assume matrix is symmetric, we use eigh in stead of eigh
    # This makes a huge difference in computation power and accuracy
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real

    sort_index = np.argsort(np.abs(eigenvalues))[::-1][:components]
    eigenvalues = eigenvalues[sort_index]
    
    W = eigenvectors[:, sort_index]

    Z = K @ W
    X_source, X_target = Z[L_s, :], Z[np.concatenate((L_d,U)), :]
    return X_source, X_target, W