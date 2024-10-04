import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from project_paths import get_results_df

#%% 
df = get_results_df("test SSTCA-V2 balanced")
df = df[df["Test Data"] == "Eval"]
df = df[df["Weighting"] == False]

df = df.drop(['Feature Extractor',
              'Version', 
              'Protocol', 
              'Sizes subsets', 
              'experiment-name', 
              'Test Data', 
              "Weighting",
              "Size L_s", 
              "Model"], axis = 1)

# Fill in the base case
df = df.fillna("NONE")

# Take mean over the random seeds
df = df.groupby(['Source Experiment', 'Target Experiment', 'Evaluation model',
                  'Size L_d', 'Size U', 'Number of components', 'Neighbours', 'Sigma', 'Lambda', 'Gamma','Mu',
                  'Train Set', 'Projection'])[["TP", "TN", "FP", "FN", "MCC"]].mean().reset_index()

df = df[df["Train Set"] != "L_s"]


#%%



for combination, group in df.groupby(['Source Experiment', 'Target Experiment', "Size L_d"]):
    plt.figure(figsize = (10,10))
    sns.boxplot(data = group,  x = "Projection", y = "MCC", hue = "Train Set")
    plt.title(combination)
    plt.show()