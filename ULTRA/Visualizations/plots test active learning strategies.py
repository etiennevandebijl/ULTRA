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

# "TP", "TN", "FP", "FN",  "Recall", "Precision", "Accuracy", "F_1", 
df = df.groupby(['Source Experiment', 'Target Experiment', 'Evaluation model',
                  'Size L_d', 'Size U', "strategy", "q", "al_model",  
                  'Train Set'])[["MCC"]].mean().reset_index()


df_MCC = pd.pivot_table(df, index = ['Evaluation model',  "al_model", "strategy", 'Train Set' ], columns = ["q", "Size L_d"], values = "MCC" )

list_pd = []
for combination, group in df.groupby(['Source Experiment', 'Target Experiment', 
       'Evaluation model', 'Size L_d', 'Size U', 'Train Set']):
    
    if group[group["strategy"] == "NONE"].shape[0] == 0:
        print(combination)
        continue
    MCC = group[group["strategy"] == "NONE"]["MCC"].values[0]
    group_ =  group[group["strategy"] != "NONE"]
    group_["MCC"] = group_["MCC"] - MCC
    list_pd.append(group_)
    
df = pd.concat(list_pd)


df_diff = pd.pivot_table(df, index = ['Evaluation model',  "al_model", "strategy", 'Train Set' ], columns = ["q", "Size L_d"], values = "MCC" )
