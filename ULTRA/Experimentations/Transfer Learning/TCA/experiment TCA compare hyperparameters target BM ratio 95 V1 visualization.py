import numpy as np
import pandas as pd

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

# Make life easier
df = df[df["l_d_size"] != 20]
df = df[df["mu"] != 10.0]
df = df.replace({"model_eval": {"NN_BF":"NN"}})

df["highest_abs_eigenvalue"] = np.log(np.abs(df["highest_abs_eigenvalue"]))
df["objective_value"] = np.log(df["objective_value"])

# Select only the eval
df_eval = df[df["test_set"] == "Eval"]

# Fill in nans with Random
df_eval[RANDOM_VARS + TCA_VARS] = df_eval[RANDOM_VARS + TCA_VARS].fillna("NONE")

# Select only training with L
df_eval_L = df_eval[df_eval["training_set"] == "L"]

# %% Compare classifiers without transformation of data

dict_ = {"L" : 'Source + Target', "L_s" : 'Source', "L_d": 'Target'}
df_eval_ = df_eval.replace({"training_set": dict_})

df_eval_ = df_eval_[df_eval_["kernel"] == "NONE"]

plot_source_target(df_eval_, hue = "training_set",
                   hue_title = "Training set",
                   plot_num_obs = False,
                   extra_info = "no TL",
                   loc_legend = (0.68, 0.92),
                   figsize = (15,7))

# %% Compare kernels when applying TCA with selecting 8 components

df_eval_L_ = df_eval_L[df_eval_L["num_components"].isin([8.0])]

plot_source_target(df_eval_L_, hue = "kernel",
                       hue_title = "TCA kernel",
                       subtitle = "mu = 1, number of components = 8, training set = Source + Target",
                       extra_info = "TL vs original mu 1 numb comp 8 training set L",
                       loc_legend = (0.75, 0.92),  
                       plot_num_obs = False)

#%% Given a classifier, what is the performance for each kernel with a certain number of components 

for model_ in ["RF", "NN"]:

    df_eval_L_ = df_eval_L[(df_eval_L["model_eval"] == model_) & (df_eval_L["kernel"] != "NONE")]
    
    plot_source_target(df_eval_L_, hue = "num_components",
                           hue_title = "TCA number of components",
                           x_var = "kernel",
                           x_label = "Kernel",                       
                           subtitle = "mu = 1, model = " + model_ + ", training set = Source + Target",
                           extra_info = "TL mu 1 model " + model_ +" training set L",
                           loc_legend = (0.72, 0.92), 
                           figsize = (19,7),
                           plot_num_obs = False)


# %% Compare classifiers performances with the number of components given the linear kernel

df_eval_L_ = df_eval_L[df_eval_L["kernel"] == "Linear"]

plot_source_target(df_eval_L_, hue = "num_components",
                       hue_title = "TCA number of components",                      
                       subtitle = "mu = 1, training set = Source + Target, kernel = Linear",
                       extra_info = "TL mu 1 training set L kernel linear",
                       loc_legend = (0.72, 0.92), 
                       figsize = (19,7),
                       plot_num_obs = False)

#%% Make Table with means over number of components with NN classifier and linear kernel

df_eval_L_ = df_eval_L[df_eval_L["kernel"] == "Linear"]
df_eval_L = df_eval_L[df_eval_L["model_eval"] == "NN"]

df_summary = pd.pivot_table(df_eval_L, index = ["source_dataset", "target_dataset"], 
                            columns = ["l_d_size", "num_components"], values = "mcc").round(3)


#%% Compare training sets 

df_eval_no_L_s = df_eval[df_eval["training_set"] != "L_s"]
df_eval_no_L_s = df_eval_no_L_s.replace({"training_set": dict_})

df_eval_no_L_s = df_eval_no_L_s[df_eval_no_L_s["kernel"].isin(["Linear"])]


for model_ in ["RF", "NN"]:

    df_eval_no_L_s_model = df_eval_no_L_s[df_eval_no_L_s["model_eval"] == model_]
    
    plot_source_target(df_eval_no_L_s_model, hue = "num_components",
                       hue_title = "TCA number of components",       
                       x_var = "training_set",
                       x_label = "Training set",  
                       subtitle = "mu = 1, model = " + model_ + ", training set = Source + Target",
                       extra_info = "TL vs original mu 1 model " + model_ +" training set L kernel linear",
                       loc_legend = (0.72, 0.92), 
                       figsize = (19,7),
                       plot_num_obs = False)


# %% Make scatterplots of eigenvalues and objective values

df_eval_L_ = df_eval_L[df_eval_L["tca_variant"] != "NONE"]

for x_var, x_label in zip(["highest_abs_eigenvalue", "objective_value"],
                          ["Max logtransformed absolute eigenvalue", "Logtransformed objective value"]):
    
    for hue, hue_title in zip(["kernel", "num_components"],
                              ["TCA kernel", "TCA number of components"]):
    
        for model_ in ["RF", "NN"]:

            df_eval_L__ = df_eval_L_[df_eval_L_["model_eval"] == model_]
            
            if hue == "num_components":
                df_eval_L__ = df_eval_L__[df_eval_L__["kernel"] == "Linear"]

            plot_source_target(df_eval_L__, hue = hue,
                               hue_title =  hue_title,
                               x_var = x_var,
                               x_label = x_label,                     
                               subtitle = "mu = 1, model = " + model_ + ", training set = Source + Target",
                               plot_type= "scatterplot",
                               extra_info = "TL vs original mu 1 model " + model_ +" training set L kernel linear")









