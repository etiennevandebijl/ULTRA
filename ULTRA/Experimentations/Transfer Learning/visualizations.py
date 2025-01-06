import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from project_paths import get_results_df

# %%
IGNORE = ['feature_extractor', 'version', 'protocol', 'uniform_sample_size', 
          'experiment_name',  "l_s_size", "train_eval_with_projection", 
          "train_eval_with_weights"]

EXP_VARS = ['source_dataset', 'target_dataset', 'l_d_size', 'u_size',
            'model_eval', 'training_set',  'test_set']

TCA_VARS = ['tca_variant', 'num_components', 'mu', 'kernel', 'semi_supervised']

SSTCA_VARS = ['num_neighbours', 'sigma', 'lambda', 'gamma', 
              'target_dependence', 'self_dependence']

RANDOM_VARS = ['random_state_subset', 'random_state_eval', 'random_state_tca'] 

EVAL_COLUMNS = ['tp', 'tn', 'fp', 'fn', 'recall', 'prec', 
                'acc', 'f1', 'mcc', 'roc_auc']

TCA_OUTCOME_VARS = ['objective_value', 'highest_abs_eigenvalue', 'sum_abs_eigenvalues']

len(EXP_VARS + TCA_VARS + SSTCA_VARS + RANDOM_VARS + EVAL_COLUMNS + TCA_OUTCOME_VARS)


# %% Download data

df_sstca = get_results_df("test SSTCA-V3 target BM ratio 95 V4")
df_tca = get_results_df("test TCA target BM ratio 95 V1")

# Remove not used variables
'''train_eval_with_weights was never true due to error in code, but that is ok'''

df_sstca = df_sstca[(df_sstca["train_eval_with_projection"] == False) & (df_sstca["train_eval_with_weights"] == False)]
df_tca = df_tca[(df_tca["train_eval_with_projection"] == False) & (df_tca["train_eval_with_weights"] == False)]

df_sstca.drop(IGNORE, axis = 1, inplace = True)
df_tca.drop(IGNORE, axis = 1, inplace = True)

# %% heatmap normal results (this picture is not really insightfull)

df_tca_ = df_tca[df_tca["test_set"] == "Eval"]
df_tca_ = df_tca_[df_tca_["training_set"] != "L_s"]

# Not selected: TCA variant, num_components, kernel, mu
# Filla in Random
df_tca_[RANDOM_VARS + TCA_VARS] = df_tca_[RANDOM_VARS + TCA_VARS].fillna("NONE")

for size, group in df_tca_.groupby(["l_d_size"]):
    
    df__ = pd.pivot_table(group, index = ["source_dataset", "target_dataset"], columns = ["model_eval", "training_set"], values = "mcc")
    plt.figure(figsize = (10,10))
    sns.heatmap(df__, annot=True)
    plt.title(size)
    plt.show()

# %% heatmap winner results

'''The question here is whether L has better results than L_D '''
for size, group in df_tca_.groupby(["l_d_size"]):
    
    if size[0] == 0:
        continue
    
    df__ = pd.pivot_table(group, index = ["source_dataset", "target_dataset", "model_eval"], columns = ["training_set"], values = "mcc")
    df__['L'] = (df__['L'] >= df__['L_d']).astype(int)
    df__['L_d'] = (df__['L'] < df__['L_d']).astype(int)
    df__ = df__.reset_index()
    df__ = pd.pivot_table(df__, index = ["source_dataset", "target_dataset"], columns = ["model_eval"], values = ["L", "L_d"])
    df__.columns = df__.columns.swaplevel(0, 1)
    df__ = df__.sort_index(axis=1)
    
    plt.figure(figsize = (10,10))
    sns.heatmap(df__, annot=True)
    plt.title(size)
    plt.show()


# %% Analyse TCA

df_tca_ = df_tca[df_tca["test_set"] == "Eval"]
df_tca_ = df_tca_[df_tca_["training_set"] == "L"]

# Filla in Random
df_tca_[RANDOM_VARS + TCA_VARS] = df_tca_[RANDOM_VARS + TCA_VARS].fillna("NONE")

# %% Compare boxplots of normal results

for source_target, df_st in df_tca_.groupby(["source_dataset", "target_dataset"]):
    
    fig, axs = plt.subplots(1, 4, figsize = (25,8))
    i = 0
    
    for size, group in df_st.groupby(["l_d_size"]):

        sns.boxplot(data = group[group["kernel"] == "NONE"], x = 'model_eval', y = "mcc", ax=axs[ i])

        axs[i].set_title("Number of labelled target instances: " + str(size[0]) )
        axs[i].set_xlabel("Evaluation model")
        axs[i].set_ylabel("MCC score trained on L and tested on (target) evaluation dataset")
        axs[i].set_ylim(-1,1)
        i = i + 1
    plt.suptitle("Source dataset: " + str(source_target[0]) + " - Target dataset : " + str(source_target[1]))
    plt.show()  

#%% Summary Table

# TCA variables 
df_tca_ = df_tca[df_tca["test_set"] == "Eval"]
df_tca_ = df_tca_[df_tca_["training_set"] == "L"]

# Filla in Random
df_tca_[RANDOM_VARS + TCA_VARS] = df_tca_[RANDOM_VARS + TCA_VARS].fillna("NONE")

df_tca_ = df_tca_[df_tca_["num_components"].isin([8.0, "NONE"])]
df_tca_ = df_tca_[df_tca_["mu"] != 10.0]

df_tca_summary = pd.pivot_table(df_tca_, index = ["kernel", "model_eval"], 
                                columns = ["source_dataset", "target_dataset","l_d_size"],
                                values = 'mcc', aggfunc = "mean") 

# %% Plot models with kernels

for source_target, df_st in df_tca_.groupby(["source_dataset", "target_dataset"]):
    
    fig, axs = plt.subplots(1, 4, figsize = (25,8))
    i = 0
    
    for size, group in df_st.groupby(["l_d_size"]):

        sns.boxplot(data = group, x = 'model_eval', y = "mcc", hue = "kernel", palette='tab10', ax=axs[ i])

        axs[i].set_title("Number of labelled target instances: " + str(size[0]) )
        axs[i].set_xlabel("Evaluation model")
        axs[i].set_ylabel("MCC score trained on L and tested on (target) evaluation dataset")
        handles, labels = axs[i].get_legend_handles_labels()
        axs[i].get_legend().remove()
        i = i + 1
    fig.legend(handles, labels, loc='upper right', ncol = 5)
    plt.suptitle("Source dataset: " + str(source_target[0]) + " - Target dataset : " + str(source_target[1]))
    plt.show()  


# %% Study effect of number of components on NN_BF

df_tca_ = df_tca[df_tca["test_set"] == "Eval"]
df_tca_ = df_tca_[df_tca_["training_set"] == "L"]

df_tca_[RANDOM_VARS + TCA_VARS] = df_tca_[RANDOM_VARS + TCA_VARS].fillna("NONE")

df_tca_ = df_tca_[df_tca_["mu"] != 10.0]
df_tca_ = df_tca_[df_tca_["model_eval"] == "RF"]

for source_target, df_st in df_tca_.groupby(["source_dataset", "target_dataset"]):
    
    fig, axs = plt.subplots(1, 4, figsize = (25,8))
    i = 0
    
    for size, group in df_st.groupby(["l_d_size"]):

        sns.boxplot(data = group, x = 'kernel', y = "mcc", hue = "num_components", palette='tab10',  ax=axs[ i])

        axs[i].set_title("Number of labelled target instances: " + str(size[0]) )
        axs[i].set_xlabel("Kernel")
        axs[i].set_ylabel("MCC score trained on L and tested on evaluation dataset")
        handles, labels = axs[i].get_legend_handles_labels()
        axs[i].get_legend().remove()
        i = i + 1
    fig.legend(handles, labels, loc='upper right', ncol = 5)
    plt.suptitle("Source dataset: " + source_target[0] + 
                 " - Target dataset : " + source_target[1])
    plt.show()  


# %%  Eigenvalue analysis

df_tca_ = df_tca[df_tca["test_set"] == "Eval"]
df_tca_ = df_tca_[df_tca_["training_set"] == "L"]

df_tca_[RANDOM_VARS + TCA_VARS] = df_tca_[RANDOM_VARS + TCA_VARS].fillna("NONE")

def scatterplot_top_eigenvalues(df, hue):
    for source_target, df_st in df.groupby(["source_dataset", "target_dataset"]):
    
        fig, axs = plt.subplots(1, 4, figsize = (25,8)); i = 0
        
        for size, group in df_st.groupby(["l_d_size"]):
    
            sns.scatterplot(x = np.log(np.abs(group["highest_abs_eigenvalue"])), y = group["mcc"], 
                            hue = group[hue], palette='tab10',  ax=axs[ i])
    
            axs[i].set_title("Number of labelled target instances: " + str(size[0]) )
            axs[i].set_xlabel("Log transformed Eigenvalue Component 1")
            axs[i].set_ylabel("MCC score model trained on L and tested on Eval")
            handles, labels = axs[i].get_legend_handles_labels()
            axs[i].get_legend().remove()
            i = i + 1
        fig.legend(handles, labels, loc='upper right', ncol = 5)
        plt.suptitle("Source dataset: " + source_target[0] + 
                     " - Target dataset : " + source_target[1])
        plt.show()    

scatterplot_top_eigenvalues(df_tca_, "kernel")
scatterplot_top_eigenvalues(df_tca_, "model_eval")
scatterplot_top_eigenvalues(df_tca_, "num_components")


scatterplot_top_eigenvalues(df_tca_[df_tca_["kernel"] != "linear"], "model_eval")


# %% Zoom in on number of components

df_tca_ = df_tca[df_tca["test_set"] == "Eval"]
df_tca_ = df_tca_[df_tca_["training_set"] == "L"]

df_tca_[RANDOM_VARS + TCA_VARS] = df_tca_[RANDOM_VARS + TCA_VARS].fillna("NONE")

df_tca_ = df_tca_[df_tca_["model_eval"] == "DT"]
df_tca_ = df_tca_[df_tca_["kernel"] == "linear"]

def boxplot_top_eigenvalues(df, x_col):
    
    for source_target, df_st in df.groupby(["source_dataset", "target_dataset"]):
        
        fig, axs = plt.subplots(1, 4, figsize = (25,8))
        i = 0
        
        for size, group in df_st.groupby(["l_d_size"]):
            
            
            sns.boxplot(x = group[x_col], y = group["mcc"], palette='tab10',  ax=axs[i])
    
            axs[i].set_title("Number of labelled target instances: " + str(size[0]) )
            axs[i].set_xlabel(x_col)
            axs[i].set_ylabel("MCC score trained on L and tested on evaluation dataset")
            handles, labels = axs[i].get_legend_handles_labels()
            i = i + 1
        fig.legend(handles, labels, loc='upper right', ncol = 5)
        plt.suptitle("Source dataset: " + source_target[0] + 
                     " - Target dataset : " + source_target[1])
        plt.show()  

boxplot_top_eigenvalues(df_tca_, "num_components")



#%% Can the L_d score say anything about Eval Score?

df_tca_ = df_tca.copy()
df_tca_["train_test_comb"] = df_tca_["training_set"] + " - " + df_tca_["test_set"]
df_tca_ = df_tca_[df_tca_["train_test_comb"].isin(["L_s - L_d", "L - Eval"])]

df_tca_ = pd.pivot_table(df_tca_, index = ['source_dataset', 'target_dataset',
                                           'l_d_size',  "model_eval", 'num_components', 
                                           'mu', 'kernel'] + RANDOM_VARS, 
                         columns = ["train_test_comb"], 
                         values= ["mcc"] )
df_tca_.columns = df_tca_.columns.droplevel()
df_tca_ = df_tca_.reset_index()

for source_target, df_st in df_tca_.groupby(["source_dataset", "target_dataset", "model_eval"]):
    
    fig, axs = plt.subplots(1, 4, figsize = (25,8))
    i = 0
    
    for size, group in df_st.groupby(["l_d_size"]):

        sns.scatterplot(x = group["L_s - L_d"], y = group["L - Eval"], hue = group["kernel"], palette='tab10',  ax=axs[ i])

        axs[i].set_title("Number of labelled target instances: " + str(size[0]) )
        axs[i].set_xlabel("MCC score L_d")
        axs[i].set_ylabel("MCC score Eval")
        handles, labels = axs[i].get_legend_handles_labels()
        i = i + 1
    fig.legend(handles, labels, loc='upper right', ncol = 5)
    plt.suptitle("Source dataset: " + source_target[0] + 
                     " - Target dataset : " + source_target[1] + 
                     " - Model :" + source_target[2])
    plt.show()   


# %% Combine datasets

df_comb = pd.concat([df_sstca, df_tca], ignore_index=True, sort=False)

# %%

df_comb_eval = df_comb[df_comb["test_set"] == "Eval"]
df_comb_eval = df_comb_eval[df_comb_eval["training_set"] == "L"]

# Filla in Random
df_comb_eval[RANDOM_VARS + TCA_VARS + SSTCA_VARS] = df_comb_eval[RANDOM_VARS + TCA_VARS + SSTCA_VARS].fillna("NONE")

# TCA variables 
df_comb_eval = df_comb_eval[df_comb_eval["num_components"].isin([8.0, "NONE"])]
df_comb_eval = df_comb_eval[df_comb_eval["kernel"].isin(["linear","NONE"])]
df_comb_eval = df_comb_eval[df_comb_eval["sigma"].isin(["1","NONE"])]

for source_target, df_st in df_comb_eval.groupby(["source_dataset", "target_dataset"]):
    
    fig, axs = plt.subplots(1, 4, figsize = (25,8)); i = 0
    
    for size, group in df_st.groupby(["l_d_size"]):

        sns.boxplot(data = group, x = 'model_eval', y = "mcc", hue = "tca_variant", palette='tab10', ax=axs[ i])

        axs[i].set_title("Number of labelled target instances: " + str(size[0]) )
        axs[i].set_xlabel("Evaluation model")
        axs[i].set_ylabel("MCC score trained on L and tested on evaluation dataset")
        handles, labels = axs[i].get_legend_handles_labels()
        axs[i].get_legend().remove()
        i = i + 1
    fig.legend(handles, labels, loc='upper right', ncol = 5)
    plt.suptitle("Source dataset: " + str(source_target[0]) + " - Target dataset : " + str(source_target[1]))
    plt.show()  

#%%
 
df_comb_eval = df_comb[df_comb["test_set"] == "Eval"]
df_comb_eval = df_comb_eval[df_comb_eval["training_set"] == "L"]

# Filla in Random
df_comb_eval[RANDOM_VARS + TCA_VARS + SSTCA_VARS] = df_comb_eval[RANDOM_VARS + TCA_VARS + SSTCA_VARS].fillna("NONE")

# TCA variables 
df_comb_eval = df_comb_eval[df_comb_eval["num_components"].isin([8.0, "NONE"])]
df_comb_eval = df_comb_eval[df_comb_eval["kernel"].isin(["linear","NONE"])]

for source_target, df_st in df_comb_eval.groupby(["source_dataset", "target_dataset"]):
    
    fig, axs = plt.subplots(1, 4, figsize = (25,8))
    i = 0
    
    for size, group in df_st.groupby(["l_d_size"]):

        sns.boxplot(data = group, x = 'lambda', y = "mcc", palette='tab10', ax=axs[ i])

        axs[i].set_title("Number of labelled target instances: " + str(size[0]) )
        axs[i].set_xlabel("Evaluation model")
        axs[i].set_ylabel("MCC score trained on L and tested on evaluation dataset")
        handles, labels = axs[i].get_legend_handles_labels()
        i = i + 1
    fig.legend(handles, labels, loc='upper right', ncol = 5)
    plt.suptitle("Source dataset: " + str(source_target[0]) + " - Target dataset : " + str(source_target[1]))
    plt.show()  


# %%

df_comb_eval = df_comb[df_comb["test_set"] == "Eval"]
df_comb_eval = df_comb_eval[df_comb_eval["training_set"] == "L"]

# Filla in Random
df_comb_eval[RANDOM_VARS + TCA_VARS + SSTCA_VARS] = df_comb_eval[RANDOM_VARS + TCA_VARS + SSTCA_VARS].fillna("NONE")

# TCA variables 
df_comb_eval = df_comb_eval[df_comb_eval["num_components"].isin([8.0, "NONE"])]
df_comb_eval = df_comb_eval[df_comb_eval["kernel"].isin(["linear","NONE"])]

# scatterplot_top_eigenvalues(df_comb_eval, "sigma")
# scatterplot_top_eigenvalues(df_comb_eval, "lambda")
# scatterplot_top_eigenvalues(df_comb_eval, "self_dependence")

#%% Boxplot eigenvalues top

boxplot_top_eigenvalues(df_comb_eval, "num_neighbours")
boxplot_top_eigenvalues(df_comb_eval, "lambda")

# %%Compare results

df_comb_eval = df_comb[df_comb["test_set"] == "Eval"]
df_comb_eval = df_comb_eval[df_comb_eval["training_set"] != "L_s"]

# Random states are not aligned:
df_comb_eval = df_comb_eval[df_comb_eval["random_state_eval"] == 0]
df_comb_eval = df_comb_eval[df_comb_eval["random_state_subset"] < 4]
df_comb_eval = df_comb_eval[df_comb_eval["model_eval"] != "SVM"]

df_comb_eval[RANDOM_VARS + TCA_VARS + SSTCA_VARS] = df_comb_eval[RANDOM_VARS + TCA_VARS + SSTCA_VARS].fillna("NONE")

# SUBSET of TCA hyperparameters
df_comb_eval = df_comb_eval[df_comb_eval["num_components"].isin([8, "NONE"])]
df_comb_eval = df_comb_eval[df_comb_eval["mu"].isin([1, "NONE"])]

# Naroow down SSTCA hyperparameters
df_comb_eval = df_comb_eval[df_comb_eval["kernel"].isin(["linear", "NONE"])]
df_comb_eval = df_comb_eval[df_comb_eval["sigma"].isin(["1", "NONE"])]
# Open = num_neighbours and lambda
df_comb_eval = df_comb_eval[df_comb_eval["num_neighbours"].isin([100, "NONE"])]
df_comb_eval = df_comb_eval[df_comb_eval["lambda"].isin([1, "NONE"])]

# Number of datapoints
df_count_datapoints = pd.pivot_table(df_comb_eval, 
                          columns = ['model_eval', "semi_supervised"], 
                          index = [ "l_d_size", 'random_state_subset'] , values = "mcc",
                          aggfunc='count').fillna(0)

highest_strategy_df_list = []
for combination, group in df_comb_eval.groupby(['source_dataset', 'target_dataset', 'l_d_size',
            'model_eval', 'random_state_subset', "training_set"]):
    index_of_interest = group['mcc'].idxmax()
    highest_strategy_df_list.append(index_of_interest)
df_selected = df_comb_eval.loc[highest_strategy_df_list]


df_count = pd.pivot_table(df_selected, 
                          columns = ['model_eval', "training_set"], 
                          index = ["l_d_size", "tca_variant", 'target_dependence',
                                   'self_dependence', "semi_supervised" ], values = "mcc",
                          aggfunc='count').fillna(0)


# %%

df_comb_eval_ = df_comb_eval[df_comb_eval["tca_variant"] != "SSTCAplus"]

highest_strategy_df_list = []

for combination, group in df_comb_eval_.groupby(['source_dataset', 'target_dataset', 'l_d_size',
            'model_eval', 'random_state_subset']):
    index_of_interest = group['mcc'].idxmax()
    highest_strategy_df_list.append(index_of_interest)
df_selected = df_comb_eval_.loc[highest_strategy_df_list]


df_count = pd.pivot_table(df_selected, 
                          columns = ['model_eval', "training_set"], 
                          index = ["l_d_size", "tca_variant"], values = "mcc",
                          aggfunc='count').fillna(0)


df_mean = pd.pivot_table(df_comb_eval_, 
                          columns = ['model_eval', "training_set"], 
                          index = ["l_d_size", "tca_variant"], values = "mcc",
                          aggfunc='mean').fillna(0)




























