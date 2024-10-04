import pandas as pd

from project_paths import get_results_df

'''
Insights: in the first iterations it is more wise to first go for random selection as certainty 
and uncertainty cannot be determined (we do not have enough data to determine these quantifications)

'''
df = get_results_df("test AL strategies")
df = df[df["Test Data"] == "Eval"]
df = df[df["Weighting"] == False]
df = df[df["Train Set"] != "L_s"]
df = df[df["Source Experiment"] != df["Target Experiment"]]

df = df.drop(['Feature Extractor',
              'Version', 
              'Protocol', 
              'Sizes subsets', 
              'experiment-name', 
              'Test Data', 
              "Weighting",
              "Size L_s", 
              "Model", 
              "Projection"], axis = 1)

df = df.fillna("NONE")

df = df.reset_index()

# %% Count

highest_strategy_df_list = []

for combination, group in df.groupby(['Source Experiment', 'Target Experiment', 'Random_states subsets',
       'Evaluation model', 'Random state eval clf', 'Size L_d', 'Size U',
        'q', 'al_model', 'Random state al clf', 'Train Set']):

    if "NONE" in group["Strategy"].values:
        continue
    
    index_of_interest = group['MCC'].idxmax()
    highest_strategy_df_list.append(index_of_interest)

df_selected = df.loc[highest_strategy_df_list]

df_count = pd.pivot_table(df_selected[df_selected['Evaluation model'] == "RF"], index = ['Evaluation model',  "al_model", "Strategy", 'Train Set' ], columns = ["q", "Size L_d"], values = "MCC", aggfunc='count')

# %%
# "TP", "TN", "FP", "FN",  "Recall", "Precision", "Accuracy", "F_1", 
df = df.groupby(['Source Experiment', 'Target Experiment', 'Evaluation model',
                  'Size L_d', 'Size U', "Strategy", "q", "al_model",  
                  'Train Set'])[["MCC"]].mean().reset_index()

df_ = df[df["al_model"] != "NONE"]
df_MCC = pd.pivot_table(df_, index = ['Evaluation model',  "al_model", "Strategy", 'Train Set' ], columns = ["q", "Size L_d"], values = "MCC" )

df__ = df_[df_["Evaluation model"] == "RF"]
df__ = df__[df__["Train Set"] == "L"]
df__ = df__[df__["al_model"] == "DT"]
df_MCC_RF = pd.pivot_table(df__, index = ["Strategy"], columns = ["q", "Size L_d"], values = "MCC" )

list_pd = []
for combination, group in df.groupby(['Source Experiment', 'Target Experiment',
       'Evaluation model', 'Size L_d', 'Size U', 'Train Set']):

    if group[group["Strategy"] == "NONE"].shape[0] == 0:
        continue
    MCC = group[group["Strategy"] == "NONE"]["MCC"].values[0]
    group_ =  group[group["Strategy"] != "NONE"]
    group_["MCC"] = group_["MCC"] - MCC
    list_pd.append(group_)

df = pd.concat(list_pd)


df_diff = pd.pivot_table(df, index = ['Evaluation model',  "al_model", "Strategy", 'Train Set' ], columns = ["q", "Size L_d"], values = "MCC" )
