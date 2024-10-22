import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from project_paths import get_results_df

# %%

df = get_results_df(".test SSTCA-V2 target BM ratio 95 V2")

df = df[df["sigma"] != "1.0"] 
df = df[df["train_eval_with_weights"] == False]

df = df.drop(['feature_extractor', 'version', 'protocol', 'uniform_sample_size', 
              'experiment_name', "train_eval_with_weights", "l_s_size"], axis = 1)

df = df[df["test_set"].isin(["L_d", "Eval"])]
df = df[df["training_set"].isin(["L_s", "L"])]

df = df[~((df["test_set"] == "Eval") & (df['training_set'] == "L_d"))]
df = df[~((df["test_set"] == "L_s") & (df['training_set'] == "L"))]

df = df[df["train_eval_with_projection"]==True]

df = df[df["model_eval"]=="DT"]

df_ = pd.pivot_table(df, index = ['source_dataset', 'target_dataset', "model_eval",
                  'l_d_size', 'u_size', 'num_components', 'num_neighbours', 'sigma', 'lambda', 'gamma', 'mu',
                  'train_eval_with_projection'], columns = "test_set", values= ["mcc"] ).reset_index()

df_.columns = df_.columns.droplevel(1)
df_.columns = ['source_dataset', 'target_dataset',"model_eval",
              'l_d_size', 'u_size', 'num_components', 'num_neighbours', 'sigma', 'lambda', 'gamma', 'mu',
              'train_eval_with_projection', "Eval", "L_d"]


for combination, group in df_.groupby(['source_dataset', 'target_dataset', 'l_d_size']):
    
    plt.figure(figsize = (10,10))
    sns.scatterplot(data = group,  x = "L_d", y = "Eval")
    plt.title(combination)
    plt.show()
