import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from project_paths import get_results_df

df = get_results_df(".test SSTCA target BM ratio 95")
df = df[df["test_set"] == "Eval"]
df = df[df["train_eval_with_weights"] == False]

df = df.drop(['feature_extractor', 'version', 'protocol', 'uniform_sample_size', 
              'experiment_name', 'test_set', "train_eval_with_weights", "l_s_size"],
             axis = 1)

df = df.fillna("NONE")

df = df.groupby(['source_dataset','target_dataset', 'model_eval',
                  'l_d_size', 'u_size', 'num_components', 'num_neighbours', 'sigma', 
                  'lambda', 'gamma', 'mu', 'training_set', 
                  'train_eval_with_projection'])[["tp", "tn", "fp", "fn", "mcc"]].mean().reset_index()

#df = df[(df["sigma"] == "NONE") | (df["train_eval_with_projection"] == True)]

df = df[df["training_set"] != "L_s"]

#df = df[df["target_dataset"] == "CIC-IDS-2018"]

df_MCC = pd.pivot_table(df, index = [ 'l_d_size', 'training_set', 'num_components', 'num_neighbours', 'sigma', 'lambda'], 
                        columns = ['gamma','mu'], values = "mcc" )


for combination, group in df.groupby(['source_dataset','target_dataset', 'l_d_size']):
    
    if combination[2] != 50:
        continue
    plt.figure(figsize = (10,10))
    sns.boxplot(data = group,  x = "train_eval_with_projection", y = "mcc", hue = "training_set")
    plt.title(combination)
    plt.show()
    

df_summary = pd.pivot_table(df[df["l_d_size"] == 20], index = ['source_dataset','target_dataset'], 
                            columns = ["training_set", "train_eval_with_projection"], 
                            values="mcc", aggfunc = "max" ).fillna(0)
