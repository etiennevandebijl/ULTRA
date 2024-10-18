import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from project_paths import get_results_df

#%% 
df = get_results_df("test SSTCA balanced")
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

# df.to_csv("/home/etienne/Dropbox/Projects/ULTRA/Results/Tables/test SSTCA balanced No Weighting Eval Test.csv", index = False )

#%% 

df = get_results_df("test SSTCA balanced No Weighting Eval Test")

# Fill in the base case
df = df.fillna("NONE")

# Take mean over the random seeds
df = df.groupby(['Source Experiment', 'Target Experiment', 'Evaluation model',
                  'Size L_d', 'Size U', 'Number of components', 'Neighbours', 'Sigma', 'Lambda', 'Gamma','Mu',
                  'Train Set', 'Projection'])[["TP", "TN", "FP", "FN", "MCC"]].mean().reset_index()

# Remove
#df = df[(df["Sigma"] == "NONE") | (df["Projection"] == True)]

df = df[df["Train Set"] != "L_s"]

# results_dict = {}
# for combination, group in df.groupby(['Source Experiment', 'Target Experiment']):
    
#     df_MCC = pd.pivot_table(group, index = [ 'Size L_d', 'Train Set', 'Number of components', 'Neighbours', 'Sigma', 'Lambda'], columns = ['Gamma','Mu'], values = "MCC" )
#     results_dict[combination] = df_MCC
    


df_summary = pd.pivot_table(df[df["Size L_d"] == 10], columns = ['Source Experiment', 'Target Experiment',  'Train Set'], index = ['Mu'], values = "MCC", aggfunc = 'mean')


df = df[df["Size L_d"] == 10]

# df = df[df['Number of components'].isin(["NONE", 4, 6, 8, 10])]

#df = df[df["Target Experiment"] == "CIC-IDS-2018"]
#%%



for combination, group in df.groupby(['Source Experiment', 'Target Experiment', ]):
    plt.figure(figsize = (10,10))
    sns.boxplot(data = group,  x = "Projection", y = "MCC", hue = "Train Set")
    plt.title(combination)
    plt.show()