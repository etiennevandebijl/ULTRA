import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from project_paths import get_results_df

df = get_results_df("test SSTCA-V2 target BM ratio 95 V2")

#df = df[df["Lambda"] > 0]
df = df[df["Sigma"] != "1.0"] 
df = df[df["Weighting"] == False]



df = df.drop(['Feature Extractor',
              'Version', 
              'Protocol', 
              'Sizes subsets', 
              'experiment-name', 
              "Weighting",
              "Size L_s"], axis = 1)


df = df[df["Test Data"].isin(["L_s"])]
df = df[df["Train Set"].isin(["L_d"])]

df = df[~((df["Test Data"] == "Eval") & (df['Train Set'] == "L_d"))]
df = df[~((df["Test Data"] == "L_s") & (df['Train Set'] == "L"))]



df = df[df["Projection"]==True]

df = df[df["Model"]=="DT"]

df_ = pd.pivot_table(df, index = ['Source Experiment', 'Target Experiment', "Model",
                  'Size L_d', 'Size U', 'Number of components', 'Neighbours', 'Sigma', 'Lambda', 'Gamma', 'Mu',
                  'Projection'], columns = "Test Data", values= ["MCC"] ).reset_index()

df_.columns = df_.columns.droplevel(1)
df_.columns = ['Source Experiment', 'Target Experiment',
                  'Size L_d', 'Size U', 'Number of components', 'Neighbours', 'Sigma', 'Lambda', 'Gamma', 'Mu',
                  'Projection', "Eval", "L_d"]


for combination, group in df_.groupby(['Source Experiment', 'Target Experiment', 'Size L_d']):
    
    plt.figure(figsize = (10,10))
    sns.scatterplot(data = group,  x = "L_d", y = "Eval")
    plt.title(combination)
    plt.show()
