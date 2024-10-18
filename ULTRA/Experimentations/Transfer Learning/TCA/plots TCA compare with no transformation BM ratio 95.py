import pandas as pd

from project_paths import get_results_df

# %% Load data

df = get_results_df("test TCA target BM ratio 95 V1")
print(df.shape) # (3665340, 42)

# Only look at evaluation data
df = df[df["Test Data"] == "Eval"]
# (733068, 42)

# Not interested in L_S
df = df[df["Train Set"] != "L_s"]
# (469152, 42)

df = df.drop(['Feature Extractor', 'Version', 'Protocol', 'Sizes subsets', 
              'experiment-name', 'Test Data', "Weighting", "Size L_s", "Size U"], axis = 1)

df = df.fillna("NONE")

# Select RF
df = df[df["Model"] == "NN_BF"]
# (234576, 42)

df_summary = pd.pivot_table(df, columns = ["Train Set","Source Experiment",
                              "Target Experiment","Size L_d"], index = ["Kernel", "TCA version", "Model","Mu"], values = 'Accuracy', aggfunc = "mean") 

df["Random_states subsets"].unique()
