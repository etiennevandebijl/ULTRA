import numpy as np
import pandas as pd

from project_paths import get_results_df

from ULTRA.visualizationTL import IGNORE, RANDOM_VARS, TCA_VARS, SSTCA_VARS, plot_source_target

# %% Loading data

df_tca = get_results_df("test TCA target BM ratio 95 V1")
df_sstca = get_results_df("test SSTCA-V3 target BM ratio 95 V4")

# Combined datasets
df = pd.concat([df_sstca, df_tca], ignore_index=True, sort=False)

df = df[(df["train_eval_with_projection"] == False) & (df["train_eval_with_weights"] == False)]

df.drop(IGNORE, axis = 1, inplace = True)

dict_ = {"linear" : 'Linear', "rbf" : 'RBF', "laplacian": 'Laplacian'}
df = df.replace({"kernel": dict_})

# Select only the eval
df_eval = df[df["test_set"] == "Eval"]

# Filla in Random
df_eval[RANDOM_VARS + TCA_VARS + SSTCA_VARS] = df_eval[RANDOM_VARS + TCA_VARS + SSTCA_VARS].fillna("NONE")


# %%

df_eval_L = df_eval[df_eval["training_set"] == "L"]

# (SS)TCA
df_eval_L = df_eval_L[df_eval_L["num_components"].isin([8.0, "NONE"])]
df_eval_L = df_eval_L[df_eval_L["mu"] != 10.0]
df_eval_L = df_eval_L[df_eval_L["kernel"].isin(["Linear", "NONE"])]
df_eval_L = df_eval_L[df_eval_L["sigma"].isin(["1","NONE"])]

df_eval_L = df_eval_L[df_eval_L["self_dependence"].isin([False,"NONE"])]
df_eval_L = df_eval_L[df_eval_L["gamma"].isin([0.5,"NONE"])]

# Not set: 'num_neighbours', 'lambda', 
    
plot_source_target(df_eval_L, hue = "tca_variant",
                       hue_title = "TCA variant",                  
                       subtitle = "mu = 1, TCA number of components = 8, sigma = 1, training set = Source + Target",
                       extra_info = "SSTCA TCA NONE mu 1 sigma 1 num comb 8 training set L",
                       experiment = "Experiment SSTCA compare hyperparameters target BM ratio 95 V1/",
                       plot_num_obs = True
                       )

plot_source_target(df_eval_L, hue = "lambda",
                       hue_title = "Lambda",                  
                       subtitle = "mu = 1, TCA number of components = 8, sigma = 1, training set = Source + Target",
                       extra_info = "SSTCA TCA NONE mu 1 sigma 1 num comb 8 training set L",
                       experiment = "Experiment SSTCA compare hyperparameters target BM ratio 95 V1/",
                       plot_num_obs = True
                       )

# %%

df_eval_L = df_eval[df_eval["training_set"] == "L"]

df_eval_L = df_eval_L[df_eval_L["tca_variant"].isin(["SSTCA", "SSTCAplus"])]
# SSTCA
df_eval_L = df_eval_L[df_eval_L["num_components"] == 8.0]
df_eval_L = df_eval_L[df_eval_L["mu"] == 1.0]
df_eval_L = df_eval_L[df_eval_L["kernel"] == "Linear"]

df_eval_L["highest_abs_eigenvalue"] = np.log(np.abs(df_eval_L["highest_abs_eigenvalue"]))
df_eval_L["objective_value"] = np.log(df_eval_L["objective_value"])

# Not defined: model, 'num_neighbours', 'lambda', 'target_dependence', 'self_dependence' sigma

plot_source_target(df_eval_L, hue = "sigma",
                       hue_title = "Sigma",  
                       x_var = "highest_abs_eigenvalue",
                       x_label = "Max logtransformed absolute eigenvalue",
                       subtitle = "mu = 1, SSTCA number of components = 8, kernel = linear, training set = Source + Target",
                       extra_info = "SSTCA mu 1 num comb 8 kernel linear training set L",
                       plot_type= "scatterplot",
                       experiment = "Experiment SSTCA compare hyperparameters target BM ratio 95 V1/"
                       )

plot_source_target(df_eval_L, hue = "lambda",
                       hue_title = "Lambda",  
                       x_var = "highest_abs_eigenvalue",
                       x_label = "Max logtransformed absolute eigenvalue",
                       subtitle = "mu = 1, SSTCA number of components = 8, kernel = linear, training set = Source + Target",
                       extra_info = "SSTCA mu 1 num comb 8 kernel linear training set L",
                       plot_type= "scatterplot",
                       experiment = "Experiment SSTCA compare hyperparameters target BM ratio 95 V1/"
                       )

plot_source_target(df_eval_L, hue = "sigma",
                       hue_title = "Sigma",  
                       x_var = "objective_value",
                       x_label = "Logtransformed objective value",
                       subtitle = "mu = 1, SSTCA number of components = 8, kernel = linear, training set = Source + Target",
                       extra_info = "SSTCA mu 1 num comb 8 kernel linear training set L",
                       plot_type= "scatterplot",
                       experiment = "Experiment SSTCA compare hyperparameters target BM ratio 95 V1/"
                       )

plot_source_target(df_eval_L, hue = "lambda",
                       hue_title = "Lambda",  
                       x_var = "objective_value",
                       x_label = "Logtransformed objective value",
                       subtitle = "mu = 1, SSTCA number of components = 8, kernel = linear, training set = Source + Target",
                       extra_info = "SSTCA mu 1 num comb 8 kernel linear training set L",
                       plot_type= "scatterplot",
                       experiment = "Experiment SSTCA compare hyperparameters target BM ratio 95 V1/"
                       )

plot_source_target(df_eval_L, hue = "sigma",
                       hue_title = "Sigma",  
                       subtitle = "mu = 1, SSTCA number of components = 8, kernel = linear, training set = Source + Target",
                       extra_info = "SSTCA mu 1 num comb 8 kernel linear training set L",
                       experiment = "Experiment SSTCA compare hyperparameters target BM ratio 95 V1/"
                       )


plot_source_target(df_eval_L, hue = "lambda",
                       hue_title = "Lambda",  
                       subtitle = "mu = 1, SSTCA number of components = 8, kernel = linear, training set = Source + Target",
                       extra_info = "SSTCA mu 1 num comb 8 kernel linear training set L",
                       experiment = "Experiment SSTCA compare hyperparameters target BM ratio 95 V1/"
                       )

plot_source_target(df_eval_L, hue = "num_neighbours",
                       hue_title = "Number of neighbours",  
                       subtitle = "mu = 1, SSTCA number of components = 8, kernel = linear, training set = Source + Target",
                       extra_info = "SSTCA mu 1 num comb 8 kernel linear training set L",
                       experiment = "Experiment SSTCA compare hyperparameters target BM ratio 95 V1/"
                       )

# %%


df_eval_L = df_eval[df_eval["training_set"] == "L"]

df_eval_L = df_eval_L[df_eval_L["model_eval"] != "SVM"]

# TCA
df_eval_L = df_eval_L[df_eval_L["num_components"].isin([8.0, "NONE"])]
df_eval_L = df_eval_L[df_eval_L["mu"].isin([1,0, "NONE"])]
df_eval_L = df_eval_L[df_eval_L["kernel"].isin(["Linear", "NONE"])]

# SSTCA
df_eval_L = df_eval_L[df_eval_L["sigma"].isin(["1", "NONE"])]
df_eval_L = df_eval_L[df_eval_L["num_neighbours"].isin([200, "NONE"])]
df_eval_L = df_eval_L[df_eval_L["lambda"].isin([1, "NONE"])]

df_eval_L = df_eval_L[df_eval_L["self_dependence"].isin([False, "NONE"])]

# Ff checken want het aantal is wel erg laag
#df_eval_L = df_eval_L[df_eval_L["random_state_eval"] == 0]
#df_eval_L = df_eval_L[df_eval_L["random_state_subset"] < 4]

plot_source_target(df_eval_L, hue = "tca_variant",
                        hue_title = "TCA variant",  
                        subtitle = "mu = 1, SSTCA number of components = 8, kernel = linear, training set = Source + Target, sigma = 1, num neigh = 100, lambda = 1",
                        extra_info = "SSTCA TCA NONE mu 1 num comb 8 kernel linear sigma 1 num neighbours 100 lambda 1 training set L",
                        experiment = "Experiment SSTCA compare hyperparameters target BM ratio 95 V1/",
                          plot_num_obs = True
                        )

for model in ["RF", "DT", "NN_BF"]:
    df_eval_L_ = df_eval_L[df_eval_L["model_eval"] == model]
    
    plot_source_target(df_eval_L_, hue = "tca_variant",
                           hue_title = "TCA variant",  
                           subtitle = "mu = 1, SSTCA number of components = 8, kernel = linear, training set = Source + Target, sigma = 1, num neigh = 100, lambda = 1 model = " + model,
                           extra_info = "SSTCA TCA NONE mu 1 num comb 8 kernel linear sigma 1 num neighbours 100 lambda 1 training set L model " + model,
                           experiment = "Experiment SSTCA compare hyperparameters target BM ratio 95 V1/",
                           plot_num_obs = True
                           )


# %%Compare results


df_eval_L = df_eval[df_eval["training_set"] != "L_s"]

df_eval_L = df_eval_L[df_eval_L["model_eval"] != "SVM"]

# TCA
df_eval_L = df_eval_L[df_eval_L["num_components"].isin([8.0, "NONE"])]
df_eval_L = df_eval_L[df_eval_L["mu"].isin([1,0, "NONE"])]
df_eval_L = df_eval_L[df_eval_L["kernel"].isin(["Linear", "NONE"])]

# SSTCA
df_eval_L = df_eval_L[df_eval_L["sigma"].isin(["1", "NONE"])]
df_eval_L = df_eval_L[df_eval_L["num_neighbours"].isin([200, "NONE"])]
df_eval_L = df_eval_L[df_eval_L["lambda"].isin([1, "NONE"])]

df_eval_L = df_eval_L[df_eval_L["self_dependence"].isin([False, "NONE"])]

df_eval_L = df_eval_L[df_eval_L["random_state_eval"] == 0]
df_eval_L = df_eval_L[df_eval_L["random_state_subset"] < 4]


# Number of datapoints
df_count_datapoints = pd.pivot_table(df_eval_L, 
                          columns = ['model_eval', "semi_supervised"], 
                          index = [ "l_d_size", 'random_state_subset'] , values = "mcc",
                          aggfunc='count').fillna(0)

highest_strategy_df_list = []
for combination, group in df_eval_L.groupby(['source_dataset', 'target_dataset', 'l_d_size',
            'model_eval', 'random_state_subset', "training_set"]):
    index_of_interest = group['mcc'].idxmax()
    highest_strategy_df_list.append(index_of_interest)
df_selected = df_eval_L.loc[highest_strategy_df_list]


df_count = pd.pivot_table(df_selected, 
                          columns = ['model_eval', "training_set"], 
                          index = ["l_d_size", "tca_variant", 'target_dependence',
                                   'self_dependence', "semi_supervised" ], values = "mcc",
                          aggfunc='count').fillna(0)


# %%

df_eval_L = df_eval_L[df_eval_L["tca_variant"] != "SSTCAplus"]

highest_strategy_df_list = []

for combination, group in df_eval_L.groupby(['source_dataset', 'target_dataset', 'l_d_size',
            'model_eval', 'random_state_subset']):
    index_of_interest = group['mcc'].idxmax()
    highest_strategy_df_list.append(index_of_interest)
df_selected = df_eval_L.loc[highest_strategy_df_list]


df_count = pd.pivot_table(df_selected, 
                          columns = ['model_eval', "training_set"], 
                          index = ["l_d_size", "tca_variant"], values = "mcc",
                          aggfunc='count').fillna(0)


df_mean = pd.pivot_table(df_eval_L, 
                          columns = ['model_eval', "training_set"], 
                          index = ["l_d_size", "tca_variant"], values = "mcc",
                          aggfunc='mean').fillna(0)





