from project_paths import get_results_df

from ULTRA.visualizationTL import EXP_VARS, TCA_VARS, RANDOM_VARS,  SSTCA_VARS

df = get_results_df("test SSTCA-V3 target BM ratio 95 V4")

counter = df.groupby(EXP_VARS + TCA_VARS + SSTCA_VARS)[RANDOM_VARS].agg("count")




