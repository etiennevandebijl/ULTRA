import pandas as pd

from project_paths import get_results_df
import matplotlib.pyplot as plt

path = "/home/etienne/Dropbox/Projects/ULTRA/Results/Figures/Experiment active learning balanced datasets/"
# %% Experiment 1 V1 Visualizations

# Revrieve results
df = get_results_df("active learning balanced datasets")
df = df[df["Test Data"] == "Eval"]
df = df[df["Weighting"] == False]
df = df[df["Use weights AL"] == False]

df = df[['Source Experiment', 'Target Experiment', 
       'Model', 'Random state clf', 'Size L_d',
       'Strategy', 'Iteration', 'Train Set', "Weighting",
       'Test Data', 'TP', 'TN', 'FP', 'FN',
       'Recall', 'Precision', 'Accuracy', 'F_1', 'MCC']]

df = df.groupby(['Source Experiment', 'Target Experiment', "Model", "Train Set", 'Size L_d',
        'Strategy', 'Iteration'])[["F_1", "MCC"]].mean().reset_index()


 # %% 
    
for combination, group in df.groupby(['Source Experiment', 'Target Experiment', "Model"]):
    
    if combination[2] == "SVM":
         continue
    
    if combination[0] == combination[1]:
        continue
    
    df_plot = pd.pivot_table(group, index = ["Size L_d"], columns = ["Strategy", "Train Set"], values = "MCC")
    
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

for combination, group in df.groupby(['Source Experiment', 'Target Experiment']):
    
    if combination[0] == combination[1]:
        continue
    
    group_ = group[group["Strategy"] == "Random"]
    group_ = group_[group_["Train Set"] != "L_s"]
    
    df_plot = pd.pivot_table(group_, index = ["Size L_d"], columns = ["Model", "Train Set"], values = "MCC")

    plt.figure(figsize = (9,9))
    df_plot.plot(title = "Source " + str(combination[0]) + " - Target " + str(combination[1]) + " - Strategy Random", figsize = (6,6))
    plt.xlabel("Number of labeled Target instances (|L_d|)")
    plt.ylabel("MCC score Eval dataset Target dataset")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=3, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(path + "AL balanced - compare models - " +"-".join(combination) + ".png")
    plt.show()
    