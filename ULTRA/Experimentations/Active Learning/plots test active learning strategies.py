import pandas as pd

from project_paths import get_results_df

'''
Insights: in the first iterations it is more wise to first go for random selection as certainty 
and uncertainty cannot be determined (we do not have enough data to determine these quantifications)

'''
df = get_results_df("test AL strategies")
df = df[df["test_set"] == "Eval"]
df = df[df["train_eval_with_weights"] == False]
df = df[df["training_set"] != "L_s"]
df = df[df["source_dataset"] != df["target_dataset"]]

df = df.drop(['feature_extractor',
              'version', 
              'protocol', 
              'uniform_sample_size', 
              'experiment_name', 
              'test_set', 
              "train_eval_with_weights",
              "l_s_size", 
              "train_eval_with_projection"], axis = 1)

df = df.fillna("NONE")

df = df.reset_index()

# %% Count

highest_strategy_df_list = []

for combination, group in df.groupby(['source_dataset', 'target_dataset', 'random_state_subset',
       'model_eval', 'random_state_eval', 'l_d_size', 'u_size',
        'query_size', 'model_al', 'random_state_al', 'training_set']):

    if "NONE" in group["al_strategy"].values:
        continue
    
    index_of_interest = group['mcc'].idxmax()
    highest_strategy_df_list.append(index_of_interest)

df_selected = df.loc[highest_strategy_df_list]

df_count = pd.pivot_table(df_selected[df_selected['model_eval'] == "RF"], 
                          index = ['model_eval',  "model_al", "al_strategy", 'training_set' ], 
                          columns = ["query_size", "l_d_size"], values = "mcc", aggfunc='count')

# %%
# "TP", "TN", "FP", "FN",  "Recall", "Precision", "Accuracy", "F_1", 
df = df.groupby(['source_dataset', 'target_dataset', 'model_eval',
                  'l_d_size', 'u_size', "al_strategy", "query_size", "model_al",  
                  'training_set'])[["mcc"]].mean().reset_index()

df_ = df[df["model_al"] != "NONE"]
df_MCC = pd.pivot_table(df_, index = ['model_eval',  "model_al", "al_strategy", 'training_set' ], 
                        columns = ["query_size", "l_d_size"], values = "mcc" )

df__ = df_[df_["model_eval"] == "RF"]
df__ = df__[df__["training_set"] == "L"]
df__ = df__[df__["model_al"] == "DT"]
df_MCC_RF = pd.pivot_table(df__, index = ["al_strategy"], columns = ["query_size", "l_d_size"], values = "mcc" )

list_pd = []
for combination, group in df.groupby(['source_dataset', 'target_dataset', 'model_eval',
                  'l_d_size', 'u_size', 'training_set']):

    if group[group["al_strategy"] == "NONE"].shape[0] == 0:
        continue
    MCC = group[group["al_strategy"] == "NONE"]["mcc"].values[0]
    group_ =  group[group["al_strategy"] != "NONE"]
    group_["mcc"] = group_["mcc"] - MCC
    list_pd.append(group_)

df = pd.concat(list_pd)


df_diff = pd.pivot_table(df, index = ['model_eval',  "model_al", "al_strategy", 'training_set' ], 
                         columns = ["query_size", "l_d_size"], values = "mcc" )
