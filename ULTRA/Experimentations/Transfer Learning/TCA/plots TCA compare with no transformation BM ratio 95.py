import pandas as pd

from project_paths import get_results_df

# %% Load data

df = get_results_df("test TCA target BM ratio 95 V1")
print(df.shape) # (3665340, 42)

# Only look at evaluation data
df = df[df["test_set"] == "Eval"]
# (733068, 42)

# Not interested in L_S
df = df[df["training_set"] != "L_s"]
# (469152, 42)

df = df.drop(['feature_extractor', 'version', 'protocol', 'uniform_sample_size', 
              'experiment_name', 'test_set', "train_eval_with_weights", "l_s_size", "u_size"], axis = 1)

df = df.fillna("NONE")

# Select RF
df = df[df["model_eval"] == "NN_BF"]
# (234576, 42)

df_summary = pd.pivot_table(df, columns = ["training_set","source_dataset",
                              "target_dataset","l_d_size"], 
                            index = ["kernel", "tca_variant", "model_eval", "mu"], 
                            values = 'acc', aggfunc = "mean") 

