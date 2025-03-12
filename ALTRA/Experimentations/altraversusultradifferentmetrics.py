import numpy as np
import pandas as pd

from project_paths import get_results_df
from DutchScaler import optimized_indicator_inverted
from tqdm import tqdm

# %% Download ALTRA and ULTRA

df_altra = get_results_df("test ALTRA target BM ratio 95 V2")
df_ultra = get_results_df("test ULTRA V2 target BM ratio 95 V3")

df = pd.concat([df_altra, df_ultra], ignore_index=True, sort=False)

#%% Make smaller

df = df.drop(['feature_extractor', 'version', 'protocol', 'uniform_sample_size',
              'query_size', "al_num_iterations", "weight_update_iterations",
              "adjust_v", "u_size", "current_iteration", "num_iterations",
              "uniform_tl_sample_size"], axis = 1)

# Select only the eval
df = df[df["test_set"] == "Eval"]

# Exclude training on Source
df = df[df["training_set"] != "L_s"]

df = df.replace({"model_eval": {"NN_BF":"NN"}})

FILLNA = ['normalize_v', 'train_eval_with_weights', 'train_eval_with_projection',
          'update_projection', 'update_weights']

df[FILLNA] = df[FILLNA].fillna("NONE")

#%% Select ULTRA and ALTRA:
    
source_target_ULTRA = (df["update_projection"] == True) & (df["training_set"] == "L") \
            & (df["update_weights"] == True) & (df["train_eval_with_projection"] == True) \
                & (df["train_eval_with_weights"] == False)

source_target_ALTRA = (df["training_set"] == "L") & (df["normalize_v"] == False)\
                    & (df["train_eval_with_weights"] == False) & (df["al_strategy"] == "Uncertainty")

df_ULTRA_ALTRA = df[source_target_ULTRA + source_target_ALTRA]

# %% Look at the performance at the end

df_ULTRA_ALTRA_ = df_ULTRA_ALTRA[(df_ULTRA_ALTRA["l_d_size"] == 100) & (df_ULTRA_ALTRA["model_eval"] == "RF")]

#%% Compute DSPI

metric = "MCC"

def return_alpha_DS(row):
    if row["mcc"] < 0:
        return 0
    y_true = np.concatenate([np.zeros(5000), np.ones(5000)])
    alpha, thetaopts = optimized_indicator_inverted(y_true, metric, row["mcc"])
    return alpha

dspi_list = []
for index, row in tqdm(df_ULTRA_ALTRA_.iterrows(), total = df_ULTRA_ALTRA_.shape[0]):
    alpha = return_alpha_DS(row)
    dspi_list.append(alpha)

df_ULTRA_ALTRA_["DSPI_MCC"] = dspi_list

#%% 

cols = ['source_dataset', 'target_dataset', 'experiment_name']

df_ULTRA_ALTRA_ = df_ULTRA_ALTRA_.replace({"experiment_name": {"test ALTRA target BM ratio 95 V2":"ALTRA",
                                     "test ultraV2 target BM ratio 95 V3":"ULTRA"}})
df_ULTRA_ALTRA__ = df_ULTRA_ALTRA_.groupby(cols)[["mcc", "DSPI_MCC"]].mean().round(3).unstack(level=-1)



































