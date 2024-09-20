import pandas as pd

from project_paths import get_results_df
import matplotlib.pyplot as plt

# %% Experiment 1 V1 Visualizations

# Revrieve results
df = get_results_df("active learning balanced datasets")
df = df[df["Test Data"] == "Eval"]
df = df[df["Weighting"] == False]
df = df[df["Use weights AL"] == False]

df = df[['Source Experiment', 'Target Experiment', 
       'Model', 'Random state clf',
       'Strategy', 'Iteration', 'Train Set', "Weighting",
       'Test Data', 'TP', 'TN', 'FP', 'FN',
       'Recall', 'Precision', 'Accuracy', 'F_1', 'MCC']]

df = df.groupby(['Source Experiment', 'Target Experiment', "Model", "Train Set",
        'Strategy', 'Iteration'])[["F_1", "MCC"]].mean().reset_index()


 # %% 
    
for combination, group in df.groupby(['Source Experiment', 'Target Experiment', "Model"]):
    
    if combination[2] == "SVM":
         continue
    
    if combination[0] == combination[1]:
        continue
    
    df_plot = pd.pivot_table(group, index = ["Iteration"], columns = ["Strategy", "Train Set"], values = "MCC")
    
    plt.figure(figsize = (9,9))
    df_plot.plot(title = str(combination))
    plt.show()
    
#%% 

for combination, group in df.groupby(['Source Experiment', 'Target Experiment', "Train Set"]):
    
    if combination[0] == combination[1]:
        continue
    
    df_plot = pd.pivot_table(group[group["Strategy"] == "Random"], index = ["Iteration"], columns = ["Model"], values = "MCC")

    plt.figure(figsize = (9,9))
    df_plot.plot(title = str(combination))
    plt.show()
    