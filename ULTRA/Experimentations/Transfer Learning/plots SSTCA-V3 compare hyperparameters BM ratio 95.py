import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from project_paths import get_results_df

# %% Load data

df = get_results_df("test SSTCA-V3 target BM ratio 95 V3")
print(df.shape) # (3665340, 42)

# Only look at evaluation data
df = df[df["Test Data"] == "Eval"]
# (733068, 42)

# Not interested in L_S
df = df[df["Train Set"] != "L_s"]
# (469152, 42)

# Select RF
df = df[df["Model"] == "RF"]
# (234576, 42)

# Exclude redudant variables
df = df.drop(['Feature Extractor', 'Version', 'Protocol', 'Sizes subsets', 
              'experiment-name', 'Test Data', "Weighting", "Model"], axis = 1)
# (234576, 34)

#df = df[df["Size L_d"] == 0]

results_dict = {}
results_dict_TCA = {}
for source_target, df_s_t in df.groupby(["Source Experiment", "Target Experiment"]):
    
    base_case = df_s_t[df_s_t["TCA version"].isnull()]
    
    base_case = pd.pivot_table(base_case, columns = ["Size L_d"], index = ["Train Set"], values = "MCC" )

    df_TCA = df_s_t[df_s_t["TCA version"] == "TCA"]
    
    df_TCA = pd.pivot_table(df_TCA, index = ["Kernel","Train Set"], columns = [ "Size L_d"], values = "MCC" )
    results_dict_TCA[source_target] = df_TCA

    df_SSTCA = df_s_t[df_s_t["TCA version"] != "TCA"]
    df_SSTCA = df_SSTCA[~df_SSTCA["TCA version"].isnull()]    

    df_SSTCA = pd.pivot_table(df_SSTCA, columns = ["Sigma", 'Train Set'], index = ["Kernel", "Size L_d","target_dependence","self_dependence"], values = "MCC" )
    results_dict[source_target] = df_SSTCA
    
