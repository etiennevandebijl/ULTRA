import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from project_paths import get_results_df

#%% 
df = get_results_df(".test SSTCA balanced")
df = df[df["test_set"] == "Eval"]
df = df[df["train_eval_with_weights"] == False]

df = df.drop(['feature_extractor',
              'version', 
              'protocol', 
              'uniform_sample_size', 
              'experiment_name', 
              'test_set', 
              "train_eval_with_weights",
              "l_s_size"], axis = 1)

# df.to_csv("/home/etienne/Dropbox/Projects/ULTRA/Results/Tables/.test SSTCA balanced No Weighting Eval Test.csv", index = False )

#%% 

df = get_results_df("test SSTCA balanced No Weighting Eval Test")

# Fill in the base case
df = df.fillna("NONE")


# Take mean over the random seeds
df = df.groupby(['source_dataset','target_dataset', 'model_eval',
                  'l_d_size', 'u_size', 'num_components', 'num_neighbours', 'sigma', 
                  'lambda', 'gamma', 'mu', 'training_set', 
                  'train_eval_with_projection'])[["tp", "tn", "fp", "fn", "mcc"]].mean().reset_index()

# Remove
#df = df[(df["sigma"] == "NONE") | (df["train_eval_with_projection"] == True)]

df = df[df["training_set"] != "L_s"]

results_dict = {}
for combination, group in df.groupby(['source_dataset','target_dataset']):
    
    df_MCC = pd.pivot_table(group, index = [ 'l_d_size', 'training_set', 'num_components', 'num_neighbours', 'sigma', 'lambda'], 
                            columns = ['gamma','mu'], values = "mcc" )
    results_dict[combination] = df_MCC
    


df_summary = pd.pivot_table(df[df["l_d_size"] == 10], 
                            columns = ['source_dataset','target_dataset', 'training_set'], 
                            index = ['mu'], values = "mcc", aggfunc = 'mean')


df = df[df["l_d_size"] == 10]

# df = df[df['num_components'].isin(["NONE", 4, 6, 8, 10])]

#df = df[df["target_dataset"] == "CIC-IDS-2018"]
#%%

for combination, group in df.groupby(['source_dataset','target_dataset']):
    plt.figure(figsize = (10,10))
    sns.boxplot(data = group,  x = "train_eval_with_projection", y = "mcc", hue = "training_set")
    plt.title(combination)
    plt.show()