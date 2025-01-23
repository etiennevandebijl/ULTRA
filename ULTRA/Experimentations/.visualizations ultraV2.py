import pandas as pd

import matplotlib.pyplot as plt

from project_paths import get_results_df

# %%
IGNORE = ['feature_extractor', 'version', 'protocol', 'uniform_sample_size', 
          'experiment_name',  "l_s_size", "train_eval_with_projection", 
          "train_eval_with_weights"]

EXP_VARS = ['source_dataset', 'target_dataset', 'l_d_size', 'u_size',
            'model_eval', 'training_set',  'test_set']

RANDOM_VARS = ['random_state_subset', 'random_state_eval', 'random_state_tca'] 

EVAL_COLUMNS = ['tp', 'tn', 'fp', 'fn', 'recall', 'prec', 
                'acc', 'f1', 'mcc', 'roc_auc']


# %% Download data

df = get_results_df("test ultraV2 target BM ratio 95 V2")

IGNORE = ['feature_extractor', 'version', 'protocol', 'uniform_sample_size', 
          'experiment_name']

df.drop(IGNORE, axis = 1, inplace = True)

# %% heatmap normal results

df = df[df["test_set"] == "Eval"]
df = df[df["training_set"] != "L_s"]
df = df[df["training_set"] != "L_d"]

IGNORE2 = ["model_al", "random_state_al", "query_size", "num_iterations", 
           "uniform_tl_sample_size", "al_strategy", "l_s_size", "u_size", 
           "test_set",'train_al_with_weights', 'train_tl_with_weights']
df.drop(IGNORE2, axis = 1, inplace = True)

IGNORE3 = ["tp","fn","fp","tn","acc","prec","recall","roc_auc","f1"]
df.drop(IGNORE3, axis = 1, inplace = True)

df = df.groupby(['source_dataset', 'target_dataset', 'update_projection', 'update_weights',
       'l_d_size', 'current_iteration', 'training_set', 'model_eval', "model_tl",
       'train_eval_with_weights', 'train_eval_with_projection'])["mcc"].mean().reset_index()

# %%
#df_ = df[df["train_eval_with_weights"] == False]
df_ = df[df["update_projection"] == True]
#df_ = df_[df_["train_eval_with_weights"] == False]

path = "/home/etienne/Dropbox/Projects/ULTRA/Results/Figures/Experiment ultra V2 bm 95/"

for comb, group in df_.groupby(['source_dataset', 'target_dataset', "model_eval"]):
    
   # if comb[2] != "RF":
   #     continue
    
    df_plot = pd.pivot_table(group, index = ["l_d_size"], columns = ["train_eval_with_weights","update_weights", "train_eval_with_projection"], values = "mcc")

    plt.figure(figsize = (9,9))
    
    df_plot.plot(title = "Source " + comb[0] + " - Target " + comb[1]  + " - " + comb[2], figsize = (6,6))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          ncol=3, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.xlabel("Number of target instances labelled")
    plt.ylabel("MCC score on evaluation dataset")
    plt.savefig(path + comb[2] +"/ultraV2 experiment - " +"-".join(comb) + ".png")
    plt.show()
    
# %%

df_ = df[df["update_projection"] == True]
df_ = df_[df_["train_eval_with_weights"] == False]

path = "/home/etienne/Dropbox/Projects/ULTRA/Results/Figures/Experiment ultra V2 bm 95/"

for comb, group in df_.groupby(['source_dataset', 'target_dataset', "model_eval"]):
    
    if comb[2] != "RF":
        continue

    df_plot = pd.pivot_table(group, index = ["l_d_size"], columns = ["update_weights", "train_eval_with_projection"], values = "mcc")

    plt.figure(figsize = (9,9))
    
    df_plot.plot(title = "Source " + comb[0] + " - Target " + comb[1]  + " - " + comb[2], figsize = (6,6))
    L = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=4, fancybox=True, shadow=True)
    L.get_texts()[0].set_text('None')    
    L.get_texts()[1].set_text('TL')
    L.get_texts()[2].set_text('NTL')    
    L.get_texts()[3].set_text('TL + NTL')
    plt.tight_layout()
    plt.xlabel("Number of target instances labelled")
    plt.ylabel("MCC score on evaluation dataset")
    plt.savefig(path + comb[2] +"/ultraV2 experiment - no train_eval_with_weights - " +"-".join(comb) + ".png")
    plt.show()    
    
# %%
for comb, group in df_.groupby(['source_dataset', 'target_dataset']):
    
    group_ = group[group["train_eval_with_weights"] == True]
    group_ = group_[group_["update_weights"] == True]
    group_ = group_[group_["train_eval_with_projection"] == True]
    
    df_plot = pd.pivot_table(group_, index = ["l_d_size"], columns = ["model_eval"], values = "mcc")

    plt.figure(figsize = (9,9))
    
    df_plot.plot(title = "Source " + comb[0] + " - Target " + comb[1], figsize = (6,6))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=3, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.xlabel("Number of target instances labelled")
    plt.ylabel("MCC score on evaluation dataset")
    plt.savefig(path +"ultraV2 experiment compare models- " +"-".join(comb) + ".png")
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    