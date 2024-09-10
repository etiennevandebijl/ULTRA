import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

from project_paths import PROJECT_PATH, get_results_df

# %% Experiment 1 V1 Visualizations

# Revrieve results
df = get_results_df("ULTRA-experiment-3")

df = df[['Source Experiment', 'Target Experiment',
        'Model', 'Random state clf',
        'Strategy', 'Size L_s', 'Size L_d', 'Size U', 'Iteration', 'Train Set',
       'Weighting', 'Projection', 'Test Data', 'TP', 'TN', 'FP', 'FN',
       'Accuracy', 'F_1', 'MCC']]

df = df[df["Train Set"] == "L"]
df = df[df["Test Data"] == "Eval"]


# df = df[df["Weighting"] == True]
# df = df[df["Projection"] == True]

# %% 

df = df.groupby(['Source Experiment', 'Target Experiment',
        'Strategy', 'Size L_d', 'Iteration', 'Weighting', 'Projection'])[["F_1", "MCC"]].mean().reset_index()

df["W and P"] = df['Weighting'].astype(str) + " " + df['Projection'].astype(str)

for combination, group in df.groupby(['Source Experiment', 'Target Experiment', "Strategy"]):
    df_plot = pd.pivot_table(group, index = ["Iteration"], columns = ["W and P"], values = "MCC")
    df_plot.plot(title = str(combination), figsize = (9,9))
    
    
for combination, group in df.groupby(['Source Experiment', 'Target Experiment']):
    group_ = group[(group["Weighting"] == True)]
    group_ = group_[group_["Projection"] == False]
    
    df_plot = pd.pivot_table(group_, index = ["Iteration"], columns = ["Strategy"], values = "MCC")
    df_plot.plot(title = str(combination), figsize = (9,9))