import numpy as np
import pandas as pd
from itertools import product

from project_paths import get_results_df

from ULTRA.visualizationTL import IGNORE, RANDOM_VARS, TCA_VARS, SSTCA_VARS, plot_source_target

# %% Loading data

df_tca = get_results_df("test TCA target BM ratio 95 V1")
df_sstca = get_results_df("test SSTCA V3 target BM ratio 95 V4")

# Combined datasets
df = pd.concat([df_sstca, df_tca], ignore_index=True, sort=False)

df = df[(df["train_eval_with_projection"] == False) & (df["train_eval_with_weights"] == False)]

df.drop(IGNORE, axis = 1, inplace = True)

# Make life easier
df = df.replace({"kernel": {"linear" : 'Linear', "rbf" : 'RBF', "laplacian": 'Laplacian'}})
df = df.replace({"sigma": {1: "1"}})

# Reduction
df = df[df["l_d_size"] != 20]
df = df.replace({"model_eval": {"NN_BF":"NN"}})

# TCA Selection: mu = 1, num_comp = 8, kernel = linear
df = df[df["mu"] != 10.0]
df = df[df["num_components"].isin([8.0, np.nan])]
df = df[df["kernel"].isin(["Linear", np.nan])]

# SSTCA Selection: no self dependence, gamma = 0.5
df = df[df["self_dependence"] != True]

df["highest_abs_eigenvalue"] = np.log(np.abs(df["highest_abs_eigenvalue"]))
df["objective_value"] = np.log(df["objective_value"])

# Select only the eval
df_eval = df[df["test_set"] == "Eval"]

# Filla in Random
df_eval[RANDOM_VARS + TCA_VARS + SSTCA_VARS] = df_eval[RANDOM_VARS + TCA_VARS + SSTCA_VARS].fillna("NONE")

df_eval_L = df_eval[df_eval["training_set"] == "L"]

conv_dict = {"highest_abs_eigenvalue": "Max logtransformed absolute eigenvalue",
             "objective_value": "Logtransformed objective value",
             "sigma": "Sigma", 
             "lambda": "Lambda", 
             "highest_abs_eigenvalue": "Max logtransformed absolute eigenvalue",
             "num_neighbours":"Number of neighbours",
             "tca_variant": "TCA variant"
              }

# %% Scatterplot comparing parameters SSTCA
    
df_eval_L_ = df_eval_L[df_eval_L["tca_variant"].isin(["SSTCA", "SSTCAplus"])]

# SSTCA
# Unselected: model, num_neighbours, lambda, sigma, target_dependence/tca_variant
for x_var, hue in product(["highest_abs_eigenvalue", "objective_value"], ["sigma", "lambda", "num_neighbours"]):

    plot_source_target(df_eval_L_, hue = hue,
                       hue_title = conv_dict[hue],  
                       x_var = x_var,
                       x_label = conv_dict[x_var],
                       subtitle = "mu = 1, SSTCA number of components = 8, kernel = linear, training set = Source + Target",
                       extra_info = "SSTCA mu 1 num comb 8 kernel linear training set L",
                       plot_type= "scatterplot",
                       experiment = "Experiment SSTCA compare hyperparameters target BM ratio 95 V1/"
                       )


# %% Boxplots comparing parameters SSTCA

df_eval_L_ = df_eval_L[df_eval_L["tca_variant"].isin(["SSTCA", "SSTCAplus"])]

# SSTCA
# Unselected: model, num_neighbours, lambda, sigma, target_dependence/tca_variant
for hue in ["sigma", "lambda", "num_neighbours", "tca_variant"]:

    plot_source_target(df_eval_L_, hue =  hue,
                       hue_title = conv_dict[hue],
                       subtitle = "mu = 1, SSTCA number of components = 8, kernel = linear, training set = Source + Target",
                       extra_info = "SSTCA mu 1 num comb 8 kernel linear training set L",
                       experiment = "Experiment SSTCA compare hyperparameters target BM ratio 95 V1/"
                       )


# %% Summary Table hyperparameters

# SSTCA
df_eval_L_ = df_eval_L[df_eval_L["sigma"].isin(["1", 1, "NONE"])]
df_eval_L_ = df_eval_L_[df_eval_L_["num_neighbours"].isin([100, "NONE"])]
df_eval_L_ = df_eval_L_[df_eval_L_["model_eval"] == "NN"]

df_summary_1 = pd.pivot_table(df_eval_L_, index = ["source_dataset", "target_dataset"], columns = ["l_d_size", "lambda"], values = "mcc").round(3)

df_eval_L_ = df_eval_L_[df_eval_L_["lambda"] == 1]

df_summary_2 = pd.pivot_table(df_eval_L_, index = ["source_dataset", "target_dataset"], columns = ["l_d_size", "sigma"], values = "mcc").round(3)



# %% boxplots Selecting Sigma = 1

# (SS)TCA
df_eval_L_ = df_eval_L[df_eval_L["tca_variant"].isin(["SSTCA", "SSTCAplus"])]
df_eval_L_ = df_eval_L_[df_eval_L_["sigma"].isin(["1", 1, "NONE"])]

# Not set: 'num_neighbours', 'lambda', 
    
for hue in ["lambda", "num_neighbours", "tca_variant"]:

    plot_source_target(df_eval_L_, hue =  hue,
                       hue_title = conv_dict[hue],
                       subtitle = "mu = 1, TCA number of components = 8, sigma = 1, training set = Source + Target",
                       extra_info = "SSTCA TCA NONE mu 1 sigma 1 num comb 8 training set L",
                       experiment = "Experiment SSTCA compare hyperparameters target BM ratio 95 V1/",
                       )


#%% Scatterplot selecting Sigma = 1

df_eval_L_ = df_eval_L[df_eval_L["tca_variant"].isin(["SSTCA", "SSTCAplus"])]
df_eval_L_ = df_eval_L_[df_eval_L_["sigma"].isin(["1", 1, "NONE"])]

# SSTCA
# Unselected: model, num_neighbours, lambda, target_dependence/tca_variant
for x_var, hue in product(["highest_abs_eigenvalue", "objective_value"], ["lambda", "num_neighbours", "model_eval"]):

    plot_source_target(df_eval_L_, hue = hue,
                       hue_title = conv_dict[hue],  
                       x_var = x_var,
                       x_label = conv_dict[x_var],
                       subtitle = "mu = 1, SSTCA number of components = 8, sigma = 1, kernel = linear, training set = Source + Target",
                       extra_info = "SSTCA mu 1 num comb 8 kernel linear training set L sigma 1",
                       plot_type = "scatterplot",
                       experiment = "Experiment SSTCA compare hyperparameters target BM ratio 95 V1/"
                       )


# %% Compare TCA variants boxplots

df_eval_L_ = df_eval_L[df_eval_L["model_eval"] != "SVM"]

# SSTCA
df_eval_L_ = df_eval_L_[df_eval_L_["sigma"].isin(["1", 1, "NONE"])]
df_eval_L_ = df_eval_L_[df_eval_L_["num_neighbours"].isin([100, "NONE"])]
df_eval_L_ = df_eval_L_[df_eval_L_["lambda"].isin([1, "NONE"])]

plot_source_target(df_eval_L_, hue = "tca_variant",
                        hue_title = "TCA variant",  
                        subtitle = "mu = 1, SSTCA number of components = 8, kernel = linear, training set = Source + Target, sigma = 1, num neigh = 100, lambda = 1",
                        extra_info = "SSTCA TCA NONE mu 1 num comb 8 kernel linear sigma 1 num neighbours 100 lambda 1 training set L",
                        experiment = "Experiment SSTCA compare hyperparameters target BM ratio 95 V1/",
                        )
summary_dict = {}
for model in ["RF", "DT", "NN"]:
    df_eval_L_model = df_eval_L_[df_eval_L_["model_eval"] == model]
    
    # Summarizing table:
    summary_dict[model] = pd.pivot_table(df_eval_L_model, index = ["source_dataset", "target_dataset"], columns = ["l_d_size", "tca_variant"], values = "mcc").round(3)

    plot_source_target(df_eval_L_model, hue = "tca_variant",
                           hue_title = "TCA variant",  
                           subtitle = "mu = 1, SSTCA number of components = 8, kernel = linear, training set = Source + Target, sigma = 1, num neigh = 100, lambda = 1 model = " + model,
                           extra_info = "SSTCA TCA NONE mu 1 num comb 8 kernel linear sigma 1 num neighbours 100 lambda 1 training set L model " + model,
                           experiment = "Experiment SSTCA compare hyperparameters target BM ratio 95 V1/",
                           )




