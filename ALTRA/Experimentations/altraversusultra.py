import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from project_paths import get_results_df

# %% Download ALTRA

df_altra = get_results_df("test ALTRA target BM ratio 95")
df_ultra = get_results_df("test ULTRA V2 target BM ratio 95 V3")

df = pd.concat([df_altra, df_ultra], ignore_index=True, sort=False)

'''
feature_extractor = NetFlow V1, version = 1_Raw, protocol = NF,
uniform_sample_size = 10000, query_size = 5, al_num_iterations = 21,
weight_update_iterations = 5, adjust_v = True, num_iterations = 20 (FML),
uniform_tl_sample_size = 1000, train_tl_with_weights = False, model_tl = "RF" (ULTRA),
model_al = DT (this is SVM for ALTRA, and random for ULTRA), v_update = True
train_al_with_weights = False (ULTRA)

'''
df = df.drop(['feature_extractor', 'version', 'protocol', 'uniform_sample_size',
              'query_size', "al_num_iterations", "weight_update_iterations",
              "adjust_v", "u_size", "current_iteration", "num_iterations",
              "uniform_tl_sample_size"], axis = 1)

df = df.replace({"model_eval": {"NN_BF":"NN"}})

# Select only the eval
df_eval = df[df["test_set"] == "Eval"]

df_eval = df_eval[df_eval["training_set"] != "L_s"]

FILLNA = ['normalize_v', 'train_eval_with_weights', 'train_eval_with_projection',
          'update_projection', 'update_weights']

df_eval[FILLNA] = df_eval[FILLNA].fillna("NONE")

cols = ['source_dataset', 'target_dataset', 'experiment_name', 'al_strategy', 
        'l_d_size', 'training_set', 'model_eval', 
        'normalize_v', 'train_eval_with_weights', 
        'train_eval_with_projection', 'update_projection', 'update_weights']

df_eval_ = df_eval.groupby(cols)["mcc"].apply(lambda x: pd.Series({'mean': x.mean(),
                                                                   'LB': x.mean() - x.std(),
                                                                   'UB': x.mean() + x.std(),
                                                                   "count": x.count()})).reset_index()

df_eval__ = pd.pivot_table(data = df_eval_, values = "mcc", index = cols, columns = "level_12").reset_index()                                                                                               


#%%
                
source_target_ULTRA = (df_eval__["update_projection"] == True) & (df_eval__["training_set"] == "L") \
            & (df_eval__["update_weights"] == True) & (df_eval__["train_eval_with_projection"] == True) \
                & (df_eval__["train_eval_with_weights"] == False)

source_target_ALTRA = (df_eval__["training_set"] == "L") & (df_eval__["normalize_v"] == False)\
                    & (df_eval__["train_eval_with_weights"] == True) & (df_eval__["al_strategy"] == "Uncertainty")


df_eval_ULTRA_ALTRA = df_eval__[source_target_ULTRA + source_target_ALTRA]

# %%

path = "/home/etienne/Dropbox/Projects/ULTRA/Results/Figures/Experiment ALTRA versus ULTRA bm 95/"

for comb, group in df_eval_ULTRA_ALTRA.groupby(['source_dataset', 'target_dataset', 'model_eval']):
                                                 
    plt.figure(figsize = (6,6))

    for name, subgroup_ in group.groupby("experiment_name"):
        
        name_ = "ALTRA"
        if "ultra" in name:
            name_ = "ULTRA"
        
        sns.lineplot( data = subgroup_, x = "l_d_size", y = "mean", label=name_)

        plt.fill_between(subgroup_["l_d_size"].values, subgroup_["LB"].values,
                         subgroup_["UB"].values, alpha=0.2)

    plt.title("Source " + comb[0] + " - Target " + comb[1] + " - Model " + comb[2])
    
    plt.legend()

    plt.xlabel("Number of target instances labelled")
    plt.ylabel("MCC score on evaluation dataset")

    plt.tight_layout()
    plt.savefig(path + comb[2] +"/ultra versus altra experiment confidence bounds - " +"-".join(comb) + ".png")
    plt.show()













































