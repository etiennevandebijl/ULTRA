import pandas as pd

import matplotlib.pyplot as plt

from project_paths import get_results_df

import seaborn as sns

# %% Download data

df = get_results_df("test ultraV2 target BM ratio 95 V3")

# Select only the eval
df_eval = df[df["test_set"] == "Eval"]

''' 
Choices:   
    'update_projection', 
    'update_weights',
    'training_set',
    'model_eval', 
    'train_eval_with_weights', 
    'train_eval_with_projection'
'''


df_eval = df_eval[df_eval["train_eval_with_weights"] == False]

df_eval = df_eval[df_eval["training_set"] != "L_s"]

cols = ['source_dataset', 'target_dataset', 'model_eval', "l_d_size", 
                 'update_projection', 'update_weights', 'training_set',
                 'train_eval_with_projection']

df_eval_ = df_eval.groupby(cols)["mcc"].apply(lambda x: pd.Series({'mean': x.mean(),
                                                                   'LB': x.mean() - x.std(),
                                                                   'UB': x.mean() + x.std(),
                                                                   "count": x.count()})).reset_index()

df_eval__ = pd.pivot_table(data = df_eval_, values = "mcc", index = cols, columns = "level_8").reset_index()                                                                                               

# %%               
path = "/home/etienne/Dropbox/Projects/ULTRA/Results/Figures/Experiment ultra V3 bm 95/"
                                                     
for comb, group in df_eval__.groupby(['source_dataset', 'target_dataset', "model_eval"]):
                                             
                                         
    base_case_indices = (group["update_projection"] == False) & (group["training_set"] == "L_d") \
                        & (group["update_weights"] == False)  & (group["train_eval_with_projection"] == False) 
    
    source_target_stacked_indices = (group["update_projection"] == False) & (group["training_set"] == "L") \
                                    & (group["update_weights"] == False) & (group["train_eval_with_projection"] == False) 
    
    source_target_SSTCA_indices = (group["update_projection"] == True) & (group["training_set"] == "L") \
                                    & (group["update_weights"] == False) & (group["train_eval_with_projection"] == True) 
    
    source_target_ULTRA_indices = (group["update_projection"] == True) & (group["training_set"] == "L") \
                                            & (group["update_weights"] == True) & (group["train_eval_with_projection"] == True) 

    plt.figure(figsize = (10,10))

    sns.lineplot( data = group[base_case_indices], x = "l_d_size", y = "mean", color = "blue", label='Target')
    sns.lineplot( data = group[source_target_stacked_indices], x = "l_d_size", y = "mean", color = "orange", label='Source + Target')
    sns.lineplot( data = group[source_target_SSTCA_indices], x = "l_d_size", y = "mean", color = "red", label = "SSTCA Source + Target")
    sns.lineplot( data = group[source_target_ULTRA_indices], x = "l_d_size", y = "mean", color = "green", label = "ULTRA")

    plt.fill_between(group[base_case_indices]["l_d_size"].values, group[base_case_indices]["LB"].values,
                     group[base_case_indices]["UB"].values, color="blue", alpha=0.2)

    plt.fill_between(group[source_target_stacked_indices]["l_d_size"].values, group[source_target_stacked_indices]["LB"].values,
                     group[source_target_stacked_indices]["UB"].values, color="orange", alpha=0.2)

    plt.fill_between(group[source_target_SSTCA_indices]["l_d_size"].values, group[source_target_SSTCA_indices]["LB"].values,
                     group[source_target_SSTCA_indices]["UB"].values, color="red", alpha=0.2)
    
    plt.fill_between(group[source_target_ULTRA_indices]["l_d_size"].values, group[source_target_ULTRA_indices]["LB"].values,
                     group[source_target_ULTRA_indices]["UB"].values, color="green", alpha=0.2)


    plt.title("Source " + comb[0] + " - Target " + comb[1]  + " - " + comb[2])
    
    plt.legend()

    plt.xlabel("Number of target instances labelled")
    plt.ylabel("MCC score on evaluation dataset")

    plt.tight_layout()
    plt.savefig(path + comb[2] +"/ultraV2 experiment confidence bounds - " +"-".join(comb) + ".png")
        
    plt.show()

#%%

path = "/home/etienne/Dropbox/Projects/ULTRA/Results/Figures/Experiment ultra V3 bm 95/"
                         
df_eval_ULTRA = df_eval__[(df_eval__["update_projection"] == True) & (df_eval__["training_set"] == "L") \
                                        & (df_eval__["update_weights"] == True) & (df_eval__["train_eval_with_projection"] == True) ]
                       
for comb, group in df_eval_ULTRA.groupby(['source_dataset', 'target_dataset']):
                                                 
    plt.figure(figsize = (10,10))

    for model, subgroup_ in group.groupby("model_eval"):
    
        sns.lineplot( data = subgroup_, x = "l_d_size", y = "mean", label=model)

        plt.fill_between(subgroup_["l_d_size"].values, subgroup_["LB"].values,
                         subgroup_["UB"].values, alpha=0.2)

    plt.title("Source " + comb[0] + " - Target " + comb[1] )
    
    plt.legend()

    plt.xlabel("Number of target instances labelled")
    plt.ylabel("MCC score on evaluation dataset")

    plt.tight_layout()
    plt.savefig(path + "/ultraV2 experiment confidence bounds - compare models - " +"-".join(comb) + ".png")
        
    plt.show()













































