import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from project_paths import get_results_df

df = get_results_df("test SSTCA-V2 target BM ratio 95")
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

df = df.fillna("NONE")

df = df.groupby(['Source Experiment', 'Target Experiment', 'Evaluation model',
                  'Size L_d', 'Size U', 'Number of components', 'Neighbours', 'Sigma', 'Lambda', 'Gamma','Mu',
                  'Train Set', 'Projection'])[["TP", "TN", "FP", "FN", "MCC"]].mean().reset_index()



df = df[df["Train Set"] != "L_s"]

#df = df[df["Target Experiment"] == "CIC-IDS-2018"]

df_MCC = pd.pivot_table(df, index = [ 'Size L_d', 'Train Set', 'Number of components', 'Neighbours', 'Sigma', 'Lambda'], columns = ['Gamma','Mu'], values = "MCC" )



df = df[df["Lambda"] != 0.0]
df = df[df["Sigma"] == "MEAN"]

for combination, group in df.groupby(['Source Experiment', 'Target Experiment', 'Size L_d']):
    
    if combination[2] != 50:
        continue
    plt.figure(figsize = (10,10))
    sns.boxplot(data = group,  x = "Projection", y = "MCC", hue = "Train Set")
    plt.title(combination)
    plt.show()
    

df_summary = pd.pivot_table(df[df["Size L_d"] == 20], index = ['Source Experiment', 'Target Experiment'], columns = ["Train Set", "Projection"], values="MCC", aggfunc = "max" ).fillna(0)
