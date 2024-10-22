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
df = df[df["test_set"] == "Eval"]
df = df[df["train_eval_with_weights"] == False]

df = df[['source_dataset', 'target_dataset', 
       'model_ultra', 'random_state_subset',
       'al_strategy', 'l_d_size', 'training_set',
       'tp', 'tn', 'fp', 'fn',
       'recall', 'prec', 'acc', 'f1', 'mcc']]

df = df.groupby(['source_dataset', 'target_dataset', 'model_ultra', "training_set",
        'al_strategy', 'l_d_size'])[['tp', 'tn', 'fp', 'fn',"f1", "mcc"]].mean().reset_index()


 # %% 
    
for combination, group in df.groupby(['source_dataset', 'target_dataset', 'model_ultra']):
    
    if combination[0] == combination[1]:
        continue
    
    df_plot = pd.pivot_table(group, index = ["l_d_size"], columns = ["al_strategy", "training_set"], values = "mcc")
    
    plt.figure(figsize = (9,9))
    df_plot.plot(title = str(combination))
    plt.show()
    

#%% 

for combination, group in df.groupby(['source_dataset', 'target_dataset', "training_set"]):

    if combination[0] == combination[1]:
        continue

    df_plot = pd.pivot_table(group[group["al_strategy"] == "Random"], index = ["l_d_size"], columns = ["model_ultra"], values = "mcc")

    plt.figure(figsize = (9,9))
    df_plot.plot(title = str(combination))
    plt.show()

