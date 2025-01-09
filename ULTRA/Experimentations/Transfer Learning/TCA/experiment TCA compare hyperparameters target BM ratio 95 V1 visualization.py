import numpy as np

from project_paths import get_results_df

from ULTRA.visualizationTL import IGNORE, RANDOM_VARS, TCA_VARS, plot_source_target

# %% Load data

# Loading
df = get_results_df("test TCA target BM ratio 95 V1")

# Remove not used variables
'''train_eval_with_weights was never true due to error in code, but that is ok'''

# Projection in this setup is in TCA variant name
df = df[(df["train_eval_with_projection"] == False) & (df["train_eval_with_weights"] == False)]

dict_ = {"linear" : 'Linear', "rbf" : 'RBF', "laplacian": 'Laplacian'}
df = df.replace({"kernel": dict_})

# Drop ignorable columns
df.drop(IGNORE, axis = 1, inplace = True)

# Select only the eval
df_eval = df[df["test_set"] == "Eval"]

# Filla in Random
df_eval[RANDOM_VARS + TCA_VARS] = df_eval[RANDOM_VARS + TCA_VARS].fillna("NONE")

# %%
dict_ = {"L" : 'Source + Target', "L_s" : 'Source', "L_d": 'Target'}
df_eval_ = df_eval.replace({"training_set": dict_})

plot_source_target(df_eval_[df_eval_["kernel"] == "NONE"], hue = "training_set",
                       hue_title = "Training set",
                       plot_num_obs = False,
                       extra_info = "no TL")

# %%

df_eval_L = df_eval[df_eval["training_set"] == "L"]
df_eval_L = df_eval_L[df_eval_L["num_components"].isin([8.0, "NONE"])]
df_eval_L = df_eval_L[df_eval_L["mu"] != 10.0]

plot_source_target(df_eval_L, hue = "kernel",
                       hue_title = "TCA kernel",
                       subtitle = "mu = 1, number of components = 8, training set = Source + Target",
                       extra_info = "TL vs original mu 1 numb comp 8 training set L",
                       plot_num_obs = False)

#%%

df_eval_L = df_eval[df_eval["training_set"] == "L"]
df_eval_L = df_eval_L[df_eval_L["mu"] != 10.0]
df_eval_L = df_eval_L[df_eval_L["model_eval"] == "RF"]

plot_source_target(df_eval_L, hue = "num_components",
                       hue_title = "TCA number of components",
                       x_var = "kernel",
                       x_label = "Kernel",                       
                       subtitle = "mu = 1, model = RF, training set = Source + Target",
                       extra_info = "TL vs original mu 1 model RF training set L",
                       plot_num_obs = False)

# %%


df_eval_L = df_eval[df_eval["training_set"] == "L"]
df_eval_L["highest_abs_eigenvalue"] = np.log(np.abs(df_eval_L["highest_abs_eigenvalue"]))
df_eval_L["objective_value"] = np.log(df_eval_L["objective_value"])
df_eval_L = df_eval_L[df_eval_L["mu"] != 10.0]
df_eval_L = df_eval_L[df_eval_L["model_eval"] == "RF"]
df_eval_L = df_eval_L[df_eval_L["tca_variant"] != "NONE"]

plot_source_target(df_eval_L, hue = "num_components",
                       hue_title = "TCA number of components",
                       x_var = "highest_abs_eigenvalue",
                       x_label = "Max logtransformed absolute eigenvalue",                       
                       subtitle = "mu = 1, model = RF, training set = Source + Target",
                       plot_type= "scatterplot",
                       extra_info = "TCA mu 1 model RF training set L log eigenvalue")

plot_source_target(df_eval_L, hue = "kernel",
                       hue_title = "TCA kernel",
                       x_var = "highest_abs_eigenvalue",
                       x_label = "Max logtransformed absolute eigenvalue",                     
                       subtitle = "mu = 1, model = RF, training set = Source + Target",
                       plot_type= "scatterplot",
                       extra_info = "TCA mu 1 model RF training set L log eigenvalue")

plot_source_target(df_eval_L, hue = "num_components",
                       hue_title = "TCA number of components",
                       x_var = "objective_value",
                       x_label = "Logtransformed objective value",                       
                       subtitle = "mu = 1, model = RF, training set = Source + Target",
                       plot_type= "scatterplot",
                       extra_info = "TCA mu 1 model RF training set L log objective value")


plot_source_target(df_eval_L, hue = "kernel",
                       hue_title = "TCA kernel",
                       x_var = "objective_value",
                       x_label = "Logtransformed objective value",                      
                       subtitle = "mu = 1, model = RF, training set = Source + Target",
                       plot_type= "scatterplot",
                       extra_info = "TCA mu 1 model RF training set L log objective value")








