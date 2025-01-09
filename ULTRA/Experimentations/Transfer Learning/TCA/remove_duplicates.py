import pandas as pd

from project_paths import get_results_df

from ULTRA.visualizationTL import IGNORE, EXP_VARS, TCA_VARS, RANDOM_VARS, EVAL_COLUMNS, TCA_OUTCOME_VARS

df = get_results_df("test TCA target BM ratio 95 V1")

df_ = df.copy()
df_.drop(IGNORE, axis = 1, inplace = True)
df_.drop(EVAL_COLUMNS, axis = 1, inplace = True)
df_.drop(TCA_OUTCOME_VARS, axis = 1, inplace = True)

duplicate_indices = df_[df_.duplicated(keep="first")].index.tolist()

df__ = df.drop(index = duplicate_indices)

df__.to_csv("/home/etienne/Dropbox/Projects/ULTRA/Results/Tables/test TCA target BM ratio 95 V1 drop.csv", index = False)





counter = df__.groupby(EXP_VARS + TCA_VARS)[RANDOM_VARS].agg("count")
