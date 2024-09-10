import numpy as np
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt

from ML.dataloader import dataloader
from ML.models import MODEL_DICT
from ML.evaluation import binary_evaluation_measures, BINARY_MEASURE_NAMES
from ML.utils import select_balanced_subset

from ML.TL.ULTRA.projection import subset_data, determine_loss_and_epsilon
from ML.TL.ULTRA.utils import normalize_weights
from ML.TL.ULTRA.SSTCAplus import SSTCAplus

#%% Load data

feature_extractor = "NetFlow V1"
version = "1_Raw"; 
protocol = "NF"; 
sizes = 10000

source_exp = "UNSW-NB15"
target_exp = "ToN-IoT"

X_s, y_s_mc, _, _ = dataloader(source_exp, feature_extractor, version, protocol, False, True)    
X_d, y_d_mc, _, _ = dataloader(target_exp, feature_extractor, version, protocol, False, True)    

X_source, y_source = select_balanced_subset(X_s, y_s_mc, sizes, 0, make_binary = True)

#X_target, y_target = select_balanced_subset(X_d, y_d_mc, sizes, 1, make_binary = True)
X_eval, y_eval = select_balanced_subset(X_d, y_d_mc, sizes, 2, make_binary= True)

selection = np.random.choice(np.where(y_d_mc == "Benign")[0], size = sizes)
X_target = X_d[selection, :]
y_target = np.zeros(sizes)

# %%

X = np.concatenate((X_source, X_target))
y = np.concatenate((y_source, y_target))

L_s = np.arange(0, X_source.shape[0])
U = np.arange(len(L_s), X.shape[0])

len_L_d = 500

L_d = np.random.choice(U, size = len_L_d)
U = np.array([u for u in U if u not in L_d])

L = np.concatenate((L_s, L_d))

w = np.ones(X.shape[0])
p = normalize_weights(w, U)

# %% Load model

model_name = "RF"
# Load ML model
clf = MODEL_DICT[model_name]


result_list = []
for rs in range(5):
    clf.set_params(random_state=rs)
    
    clf.fit(X = X[L], y = y[L])
    y_pred = clf.predict(X_eval)
    results = binary_evaluation_measures(y_eval, y_pred)
    result_list.append(results)

summary  = np.array(result_list).mean(axis = 0)
dictionary = dict(zip(BINARY_MEASURE_NAMES, summary))

# %% SSTCAplus


subset_size = 1000

S_T_t, L_s_subset, L_d_subset, U_subset = subset_data(L_s, L_d, U, p, subset_size)

results_list = []

for rs in range(10):
    print(rs)
    for c, k, sigma, lamda, gamma, mu in product([2, 4, 10], 
                                             [10, 50, 100], #k 
                                             ["MED"], # Sigma
                                             [10, 100, 1000], # Lambda (Study)
                                             [0.5], # Gamma (study) #BALANCING
                                             [1.0] #mu (default original) 
                                             ):
        # We construct a possible projection matrix B
        _, _, B = SSTCAplus(X[S_T_t], y[S_T_t], L_s_subset, L_d_subset, U_subset, 
                            components = c, 
                            k = k, 
                            sigma = sigma,
                            lamda = lamda,
                            gamma = gamma,
                            mu = mu)
    
        # As we selected Linear, we can simply circomvent the kernel
        B = X[S_T_t].T @ B

        eps_B, loss = determine_loss_and_epsilon(X, y, L, L_s, L_d, B, w, p, model_name, rs)

        clf.fit(X = X[L] @ B, y = y[L])
        y_pred = clf.predict(X_eval @ B)
        results = binary_evaluation_measures(y_eval, y_pred)

        results_list.append([c, k, sigma, lamda, gamma, mu, eps_B, results[8]])
        
        df = pd.DataFrame(results_list, columns = ["c", "k", "sig", "lam", "gam", "mu", "eps_B", "mcc"])
        
        df_ = df.groupby(["c", "k", "sig", "lam", "gam", "mu"])[["eps_B", "mcc"]].mean().reset_index()
        #df_ = df_[df_["c"] == 10]
        plt.figure(figsize = (10,10))
        plt.scatter(df_["eps_B"], df_["mcc"])
        plt.xlabel("Epsilon")
        plt.ylabel("MCC")
        plt.axhline(y = dictionary["MCC"], color = "red")
        plt.show()        









