import pandas as pd

from project_paths import get_results_df
import matplotlib.pyplot as plt

path = "/home/etienne/Dropbox/Projects/ULTRA/Results/Figures/Experiment active learning balanced datasets/"
# %% Experiment 1 V1 Visualizations

# Revrieve results
df = get_results_df("active learning balanced datasets")
df = df[df["test_set"] == "Eval"]
df = df[df["train_eval_with_weights"] == False]
df = df[df["train_al_with_weights AL"] == False]

df = df[['source_dataset', 'target_dataset', 
       'model_ultra', 'random_state_eval', 'l_d_size',
       'al_strategy', 'current_iteration', 'training_set', "train_eval_with_weights",
       'test_set', 'tp', 'tn', 'fp', 'fn',
       'recall', 'prec', 'acc', 'f1', 'mcc']]

df = df.groupby(['source_dataset', 'target_dataset','model_ultra',"training_set", 'l_d_size',
        'al_strategy', 'current_iteration'])[["f1", "mcc"]].mean().reset_index()


 # %% 
    
for combination, group in df.groupby(['source_dataset', 'target_dataset','model_ultra']):
    
    if combination[2] == "SVM":
         continue
    
    if combination[0] == combination[1]:
        continue
    
    df_plot = pd.pivot_table(group, index = ["l_d_size"], columns = ["al_strategy", "training_set"], values = "mcc")
    
    plt.figure(figsize = (9,14))
    df_plot.plot(title = "Source " + str(combination[0]) + " - Target " + str(combination[1]) + " - Model " + str(combination[2]), figsize = (6,6))
    plt.xlabel("Number of labeled Target instances (|L_d|)")
    plt.ylabel("MCC score Eval dataset Target dataset")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          ncol=3, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(path + "AL balanced " +"-".join(combination) + ".png")
    plt.show()
    
#%% 

for combination, group in df.groupby(['source_dataset', 'target_dataset']):
    
    if combination[0] == combination[1]:
        continue
    
    group_ = group[group["al_strategy"] == "Random"]
    group_ = group_[group_["training_set"] != "L_s"]
    
    df_plot = pd.pivot_table(group_, index = ["l_d_size"], columns = ["model_ultra", "training_set"], values = "mcc")

    plt.figure(figsize = (9,9))
    df_plot.plot(title = "Source " + str(combination[0]) + " - Target " + str(combination[1]) + " - Strategy Random", figsize = (6,6))
    plt.xlabel("Number of labeled Target instances (|L_d|)")
    plt.ylabel("MCC score Eval dataset Target dataset")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=3, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(path + "AL balanced - compare models - " +"-".join(combination) + ".png")
    plt.show()
    