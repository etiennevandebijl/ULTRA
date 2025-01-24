import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from project_paths import get_results_df

# %% Download data

df = get_results_df("test ULTRA V2 target BM ratio 95 V3")
df = df.replace({"model_eval": {"NN_BF":"NN"}})

# Select only the eval
df_eval = df[df["test_set"] == "Eval"]

df_eval = df_eval[df_eval["train_eval_with_weights"] == False]

cols = ['source_dataset', 'target_dataset', 'model_eval', "l_d_size", 'update_projection', 
        'update_weights', 'training_set', 'train_eval_with_projection']

df_eval = df_eval.groupby(cols)["mcc"].apply(lambda x: pd.Series({'mean': x.mean(),
                                                                   'LB': x.mean() - x.std(),
                                                                   'UB': x.mean() + x.std(),
                                                                   "count": x.count()})).reset_index()

df_eval = pd.pivot_table(data = df_eval, values = "mcc", index = cols, columns = "level_8").reset_index()                                                                                               

# %%

df_eval["label"] = "NONE"

base_case = (df_eval["update_projection"] == False) & (df_eval["training_set"] == "L_d") \
            & (df_eval["update_weights"] == False)  & (df_eval["train_eval_with_projection"] == False) 
df_eval.loc[base_case, "label"] = "Target"

source_target = (df_eval["update_projection"] == False) & (df_eval["training_set"] == "L") \
            & (df_eval["update_weights"] == False) & (df_eval["train_eval_with_projection"] == False) 
df_eval.loc[source_target, "label"] = "Source + Target"

source_target_SSTCA = (df_eval["update_projection"] == True) & (df_eval["training_set"] == "L") \
            & (df_eval["update_weights"] == False) & (df_eval["train_eval_with_projection"] == True) 
df_eval.loc[source_target_SSTCA, "label"] = "SSTCA Source + Target"

source_target_ULTRA = (df_eval["update_projection"] == True) & (df_eval["training_set"] == "L") \
            & (df_eval["update_weights"] == True) & (df_eval["train_eval_with_projection"] == True) 
df_eval.loc[source_target_ULTRA, "label"] = "ULTRA Source + Target"

colour_dict = {"Target":"blue",
               "Source + Target": "orange",
               "SSTCA Source + Target": "red",
               "ULTRA Source + Target": "green"}

df_eval = df_eval[df_eval["label"] != "NONE"]


# %%               
path = "/home/etienne/Dropbox/Projects/ULTRA/Results/Figures/Experiment ULTRA V3 bm 95/"
                                                     
for comb, df_eval_st in df_eval.groupby(['source_dataset', 'target_dataset', "model_eval"]):

    plt.figure(figsize = (6,6))

    for label, df_eval_st_lab in df_eval_st.groupby(["label"]):

        sns.lineplot(data = df_eval_st_lab, x = "l_d_size", y = "mean", color = colour_dict[label[0]], label=label)

        plt.fill_between(df_eval_st_lab["l_d_size"].values, df_eval_st_lab["LB"].values,
                         df_eval_st_lab["UB"].values, color=colour_dict[label[0]], alpha=0.2)

    plt.title("Source " + comb[0] + " - Target " + comb[1]  + " - " + comb[2])
    
    plt.legend()

    plt.xlabel("Number of target instances labeled")
    plt.ylabel("MCC score on evaluation dataset")

    plt.tight_layout()
    plt.savefig(path + comb[2] +"/ultraV3 experiment confidence bounds - " +"-".join(comb) + ".png")
        
    plt.show()

#%%
                         
df_eval_ULTRA = df_eval[source_target_ULTRA]
                       
for comb, group in df_eval_ULTRA.groupby(['source_dataset', 'target_dataset']):
                                                 
    plt.figure(figsize = (6,6))

    for model, subgroup_ in group.groupby("model_eval"):
    
        sns.lineplot(data = subgroup_, x = "l_d_size", y = "mean", label=model)

        plt.fill_between(subgroup_["l_d_size"].values, subgroup_["LB"].values,
                         subgroup_["UB"].values, alpha=0.2)

    plt.title("Source " + comb[0] + " - Target " + comb[1] )
    
    plt.legend()

    plt.xlabel("Number of target instances labeled")
    plt.ylabel("MCC score on evaluation dataset")

    plt.tight_layout()
    plt.savefig(path + "/ultraV3 experiment confidence bounds - compare models - " +"-".join(comb) + ".png")
        
    plt.show()





