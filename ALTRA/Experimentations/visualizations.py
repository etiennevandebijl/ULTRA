import pandas as pd

import matplotlib.pyplot as plt

from project_paths import get_results_df

import seaborn as sns

# %% Download data

df = get_results_df("test altra target BM ratio 95")

# Select only the eval
df_eval = df[df["test_set"] == "Eval"]

''' 
Choices:   

'''

df_eval = df_eval[df_eval["train_eval_with_weights"] == True]

df_eval = df_eval[df_eval["training_set"] != "L_s"]

cols = ['source_dataset', 'target_dataset', 'model_eval', "l_d_size", 
                 'al_strategy','normalize_v']

df_eval_ = df_eval.groupby(cols)["mcc"].apply(lambda x: pd.Series({'mean': x.mean(),
                                                                   'LB': x.mean() - x.std(),
                                                                   'UB': x.mean() + x.std(),
                                                                   "count": x.count()})).reset_index()

df_eval__ = pd.pivot_table(data = df_eval_, values = "mcc", index = cols, columns = "level_6").reset_index()                                                                                               


#%%
                
                       
for comb, group in df_eval__.groupby(['source_dataset', 'target_dataset']):
                                                 
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

    plt.show()













































