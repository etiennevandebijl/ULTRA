import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from project_paths import get_results_df

# %% Load data

df = get_results_df(".test SSTCA-V2 target BM ratio 95 V2")
print(df.shape) # (753060, 35)

# Only look at evaluation data
df = df[df["test_set"] == "Eval"]

#No Weighting
df = df[df["train_eval_with_weights"] == False]

# Not interested in L_S
df = df[df["training_set"] != "L_s"]

# Exclude redudant variables

# Select RF
df = df[df["model_eval"] == "RF"]

df = df.drop(['feature_extractor', 'version', 'protocol', 'uniform_sample_size', 
              'experiment_name', 'test_set', "train_eval_with_weights"], axis = 1)

# Variables SSTCA: 'Number of components', 'Neighbours', 'Sigma', 'Lambda', 'Gamma', 'Mu'
# Evaluation: Top eigenvalue, MCC

# RF does automatic feature-selection, so lets keep the original data space
# We can always look at reducing the dataspace, but results are not yet decisive.
df = df[df["num_components"] == 8]
df.shape


#%% 
cols = ['l_d_size', 'num_components', 'num_neighbours', 'sigma', 'lambda', 
        'gamma', 'mu', 'training_set', 'train_eval_with_projection']

for source_target, df_st in df.groupby(["source_dataset", "target_dataset"]):
    
    fig, axs = plt.subplots(1, 4, figsize = (25,8))
    i = 0
    
    min_MCC = df_st.groupby(cols)[["mcc"]].mean().min()[0]
    max_MCC = df_st.groupby(cols)[["mcc"]].mean().max()[0]
    
    for size, group in df_st.groupby(["l_d_size"]):

        group_B = group[(group["train_eval_with_projection"]==False)].groupby(["training_set"])["mcc"].mean().to_dict()
        
        group_P = group[(group["train_eval_with_projection"]==True) & (group['training_set'] == "L")]
        
        group_P = group_P.groupby(cols)[["mcc", "highest_abs_eigenvalue", "sum_abs_eigenvalues"]].mean().reset_index()
        
        
        colours = {"L": "black", "L_d": "red"}
        for TS, mcc in group_B.items():
            axs[i].axhline(mcc, label = "MCC RF trained on " + str(TS), color = colours[TS])
        
        sns.scatterplot(x =np.log(np.abs(group_P["highest_abs_eigenvalue"])), y = group_P["mcc"], hue = group_P["sigma"], palette='tab10',  ax=axs[ i])

        axs[i].set_title("Number of labelled target instances: " + str(size[0]) )
        axs[i].set_xlabel("Log transformed Eigenvalue Component 1")
        axs[i].set_ylabel("MCC score Random Forest trained on L and tested on evaluation dataset")
        handles, labels = axs[i].get_legend_handles_labels()
        axs[i].get_legend().remove()
        axs[i].set_ylim(min_MCC,max_MCC)
        i = i + 1
    fig.legend(handles, labels, loc='upper right', ncol = 5)
    plt.suptitle("Source dataset: " + str(source_target[0]) + " - Target dataset : " + str(source_target[1]))
    plt.show()    

# %%

for source_target, df_st in df.groupby(["source_dataset", "target_dataset"]):
    

    fig, axs = plt.subplots(1, 4, figsize = (25,8))
    i = 0
    

    for size, group in df_st.groupby(["l_d_size"]):

        group_B = group[(group["train_eval_with_projection"]==False)].groupby(["training_set"])["mcc"].mean().to_dict()
        
        group_P = group[(group["train_eval_with_projection"]==True) & (group['training_set'] == "L")]


        sns.boxplot(x =group_P["lambda"], y = group_P["mcc"], hue = group_P["num_neighbours"], palette='tab10',  ax=axs[ i])
        handles, labels = axs[i].get_legend_handles_labels()
        axs[i].get_legend().remove()
        axs[i].set_ylim(min_MCC,max_MCC)
        i = i + 1
    fig.legend(handles, labels, loc='upper right', ncol = 5)
    plt.suptitle("Source dataset: " + str(source_target[0]) + " - Target dataset : " + str(source_target[1]))
    plt.show()  




