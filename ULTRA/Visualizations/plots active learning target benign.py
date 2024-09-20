import pandas as pd

from project_paths import get_results_df
import matplotlib.pyplot as plt

'''
The conclusion of this experiment is that by introducing only a little bit of 
benign traffic of the target dataset, the model tends to only predict Benign.
Why? we now have a notion of the target data distribution and its traffic 
both normal/malicious will be in the same direction. The more normal traffic
we will include, the more it will predict everything benign. This is for both V1 
where q = 50 as well as q=900, where this will even be more extreme. 

'''
# %% Experiment 1 V1 Visualizations

# Revrieve results
df = get_results_df("active learning target benign V2")
df = df[df["Test Data"] == "Eval"]
df = df[df["Weighting"] == False]

df = df[['Source Experiment', 'Target Experiment', 
       'Model', 'Random state clf',
       'Strategy', 'Size L_d', 'Train Set', "Weighting",
       'Test Data', 'TP', 'TN', 'FP', 'FN',
       'Recall', 'Precision', 'Accuracy', 'F_1', 'MCC']]

df = df.groupby(['Source Experiment', 'Target Experiment', "Model", "Train Set",
        'Strategy', 'Size L_d'])[['TP', 'TN', 'FP', 'FN',"F_1", "MCC"]].mean().reset_index()


 # %% 
    
for combination, group in df.groupby(['Source Experiment', 'Target Experiment', "Model"]):
    
    if combination[0] == combination[1]:
        continue
    
    df_plot = pd.pivot_table(group, index = ["Size L_d"], columns = ["Strategy", "Train Set"], values = "MCC")
    
    plt.figure(figsize = (9,9))
    df_plot.plot(title = str(combination))
    plt.show()
    

#%% 

for combination, group in df.groupby(['Source Experiment', 'Target Experiment', "Train Set"]):

    if combination[0] == combination[1]:
        continue

    df_plot = pd.pivot_table(group[group["Strategy"] == "Random"], index = ["Size L_d"], columns = ["Model"], values = "MCC")

    plt.figure(figsize = (9,9))
    df_plot.plot(title = str(combination))
    plt.show()

