import numpy as np

from sklearn.utils import check_array
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import euclidean_distances, pairwise_kernels

from scipy import linalg

''' Example data
y = np.random.choice([0,1], size = 100)
X = np.random.uniform(size = (100,10))
M_total = len(y)
L_s = np.arange(50)
L_d = np.arange(50, 70)
U = np.arange(70, 100)
M_source = len(L_s)
M_target = M_total - M_source
'''

def centering_matrix_H(M_total):
    return np.eye(M_total) - (1 / M_total) * np.ones((M_total, M_total))
    
def coefficient_matrix_L(M_source, M_target, L_s):
    if len(L_s) != M_source:
        raise ValueError("M_source should be equal to len(L_s)")
        
    e = np.ones(( M_source + M_target, 1)) / M_target
    e[L_s, 0] = -1 / M_source
    L = e * e.T
    return L

def label_dependence_K_yy_star(y, U, L_d, gamma = 0.5, target_dependence = False, self_dependence = True):
    M_total = len(y)
    
    Kyy = (y.reshape(-1, 1) == y).astype(int)    
    
    #Exclude non-labelled
    a = np.ones((M_total, 1))
    a[U, 0] = 0 #DIT SWS want we weten we niet

    if target_dependence:
        a[L_d, 0] = 0
        
    O = a * a.T
    
    # Moet ik de diagonal niet op 1 zetten?    
    if self_dependence:
        O = np.maximum(O, np.eye(M_total))
    
    Kyy = np.multiply(Kyy, O)

    Kyy_star = gamma * Kyy + (1 - gamma) * np.eye(M_total)
    return Kyy_star
    
def locality_preserving(X, k: int = 100, sigma = 1.0):
    M_total = X.shape[0]
        
    # locality preservation
    # https://en.wikipedia.org/wiki/Euclidean_distance
    dist_matrix = euclidean_distances(X)
    
    knn_graph = kneighbors_graph(X, n_neighbors = k, 
                                 mode='connectivity', include_self=False).toarray()
    
    # Make the graph symmetric
    knn_graph = np.maximum(knn_graph, knn_graph.T)
    
    # Semisupervised TCA for DA in Remote Sensing
    if sigma == "MED":
        sigma = np.median(dist_matrix)
    
    #https://ieeexplore-ieee-org.vu-nl.idm.oclc.org/stamp/stamp.jsp?tp=&arnumber=5570959
    if sigma == "MEAN":
        sigma = np.mean(dist_matrix)
        
    # Affinity/adjecency  matrix (RBF ?)
    M = np.exp(- dist_matrix**2 / (2 * sigma**2)) * knn_graph
    
    # Degree matrix ()
    D = np.sum(M, axis = 0) * np.eye(M_total)
    
    Laplace = D - M
    return Laplace
    

def SSTCAplusV3(X, y, L_s, L_d, U, components: int = 8, k: int = 100, sigma = 1.0, lamda = 1, kernel = "linear",
              gamma: float = 0.5, mu: float = 1., random_state: int = 0,
              semi_supervised = True, target_dependence = False, self_dependence = True):
    
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
    # If linear, it simple computes X @ X.T
    
    # Coefficient Matrix
    L = coefficient_matrix_L(M_source, M_target, L_s)

    # Construct centering matrix H
    H = centering_matrix_H(M_total)

    if semi_supervised:
        # maximize the dependence between the extracted features and the class labels
        Kyy_star = label_dependence_K_yy_star(y, U, L_d, gamma, target_dependence, self_dependence)
    
        # Local Preservity
        Laplace = locality_preserving(X, k, sigma)
    
        # This is more computationally stable
        a = K @ (L + lamda * Laplace) @ K + mu * np.eye(M_total)
        b = K @ H @ Kyy_star @ H @ K
    else:
        a = K @ L @ K + mu * np.eye(M_total)
        b = K @ H @ K
        
    # Compute matrix
   # print(a.shape)
    # matrix = np.linalg.inv(a) @ b
    matrix = linalg.lstsq(a, b)[0]

    # Retrieve eigenvectors and eigenvalues (matrix is not symmetric like before)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    # Make the values real
    eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real

    # Sort eigenvalues
    sort_index = np.argsort(np.abs(eigenvalues))[::-1][:components]
    
    # Get eigenvalues in order
    eigenvalues = eigenvalues[sort_index]
   # print(eigenvalues)
    
    # Projection matrix
    W = eigenvectors[:, sort_index]
    
    # Compute the constraint
    # constraint = W.T @ b @ W
    # print(constraint)
    
    # Compute the objective 
    obj_1 = np.trace(W.T @ K @ L @ K @ W)
    obj_2 = mu * np.trace(W.T @ W)
    obj_3 = 0
    if semi_supervised:
        obj_3 = lamda * np.trace(W.T @ K @ Laplace @ K @ W)
    obj = obj_1 + obj_2 + obj_3
    # print(obj)
    
    # Compute the projected data ( I am not sure about this one)
    # Z = K @ W
    
    # X_source, X_target = Z[L_s, :], Z[np.concatenate((L_d,U)), :]
    
    return W, eigenvalues, obj


def project_data(X_new, X, W, kernel):
    K = pairwise_kernels(X_new, X, metric = kernel)
    return K @ W
