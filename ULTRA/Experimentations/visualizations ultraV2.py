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

df = get_results_df("test ultraV2 target BM ratio 95 V1")

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
df_ = df_[df_["train_eval_with_weights"] == False]

path = "/home/etienne/Dropbox/Projects/ULTRA/Results/Figures/Experiment ultra V2 bm 95/"

for comb, group in df_.groupby(['source_dataset', 'target_dataset',"model_eval"]):
    
    
    df_plot = pd.pivot_table(group, index = ["l_d_size"], columns = [ "training_set", "update_weights", "train_eval_with_projection"], values = "mcc")


    plt.figure(figsize = (9,9))
    
    df_plot.plot(title = "Source " + comb[0] + " - Target " + comb[1]  + " - " + comb[2], figsize = (6,6))

    plt.tight_layout()
    plt.savefig(path + "ultraV2 experiment - " +"-".join(comb) + ".png")
    plt.show()
    