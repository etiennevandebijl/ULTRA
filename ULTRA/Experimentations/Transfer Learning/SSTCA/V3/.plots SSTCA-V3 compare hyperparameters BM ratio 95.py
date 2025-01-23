import pandas as pd

from project_paths import get_results_df

# %% Load data

df = get_results_df("test SSTCA-V3 target BM ratio 95 V4")
print(df.shape) 

# Only look at evaluation data
df = df[df["test_set"] == "Eval"]

df = df[df["training_set"] != "L_s"]

# Exclude redudant variables
df = df.drop(['feature_extractor', 'version', 'protocol', 'uniform_sample_size', 
              'experiment_name', 'test_set', "train_eval_with_weights"], axis = 1)
#df = df[df["Size L_d"] == 0]

results_dict = {}
results_dict_TCA = {}
for source_target, df_s_t in df.groupby(["source_dataset", "target_dataset"]):
    
    base_case = df_s_t[df_s_t["tca_variant"].isnull()]
    
    base_case = pd.pivot_table(base_case, columns = ["l_d_size"], index = ["training_set"], values = "mcc" )

    df_TCA = df_s_t[df_s_t["tca_variant"] == "TCA"]
    
    df_TCA = pd.pivot_table(df_TCA, index = ["kernel","training_set"], columns = ["l_d_size"], values = "mcc" )
    results_dict_TCA[source_target] = df_TCA

    df_SSTCA = df_s_t[df_s_t["tca_variant"] != "TCA"]
    df_SSTCA = df_SSTCA[~df_SSTCA["tca_variant"].isnull()]    

    df_SSTCA = pd.pivot_table(df_SSTCA, columns = ["sigma", 'training_set'], 
                              index = ["kernel", "l_d_size","target_dependence","self_dependence"], 
                              values = "mcc" )
    results_dict[source_target] = df_SSTCA
    
