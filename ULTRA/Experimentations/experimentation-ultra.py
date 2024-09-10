import numpy as np
import pandas as pd
from tqdm import tqdm

from ML.dataloader import dataloader
from ML.TL.ULTRA.ultra import ultra
from ML.models import MODEL_DICT
from ML.utils import select_balanced_subset

import warnings
warnings.filterwarnings("ignore")

#%% Load data

source_exp = "UNSW-NB15"; target_exp = "BoT-IoT"; feature_extractor = "NetFlow V1"
version = "1_Raw"; protocol = "NF"; model_name = "RF"

X_source, y_source_mc, _, _ = dataloader(source_exp, feature_extractor, version, protocol, False, True) 
X_target, y_target_mc, _, _ = dataloader(target_exp, feature_extractor, version, protocol, False, True)

novel_attack = "DDoS"
target_instances = [y_i != novel_attack  for y_i in y_target_mc]
test_instances = [y_i == novel_attack or y_i == "Benign" for y_i in y_target_mc]

SIZE = 1000

results_list = []
for rs in tqdm(range(1, 30, 3)):

    X_source_ss, y_source_ss = select_balanced_subset(X_source, y_source_mc, 100, rs)
    X_target_ss, y_target_ss = select_balanced_subset(X_target[target_instances], y_target_mc[target_instances], 100, rs + 1)
    X_target_test_ss, y_target_test_ss = select_balanced_subset(X_target[test_instances], y_target_mc[test_instances], SIZE, rs + 2)
    
    clf = MODEL_DICT[model_name]

    for strategy in ["Random"]:
        clf, results = ultra(X_source_ss, y_source_ss, X_target_ss, y_target_ss,
                             X_target_test_ss, y_target_test_ss, clf, sn = 1, sp=0.5, iter_ = 10, strategy = strategy)    
        results = [[rs, strategy] + x for x in results]
        results_list.append(np.array(results))

# %% Process results

results_array = np.concatenate(results_list, axis = 0)
df = pd.DataFrame(results_array, columns = ["RS", "Strategy", "Labelled", "Only Target", "Target and Source", "Target Souce and TradaBoost"])

df[["Labelled"]] = df[["Labelled"]].astype(int)
df[["Only Target", "Target and Source", "Target Souce and TradaBoost"]] = df[["Only Target", "Target and Source", "Target Souce and TradaBoost"]].astype(float)

# %% Visualizations

df_ = df[df["Strategy"] == "Random"].groupby(["Labelled"])["Only Target","Target and Source", "Target Souce and TradaBoost"].mean()
df_.plot(figsize = (10,10), ylabel ="MCC score on Test dataset", xlabel = "Number of labelled instances", title = "Effect TrAdaBoost on Random Sampling")

df_ = df[df["Strategy"] == "Random"].groupby(["Labelled"])["Target and Source", "Target Souce and TradaBoost"].mean()
df_.plot(figsize = (10,10), ylabel ="MCC score on Test dataset", xlabel = "Number of labelled instances", title = "Use all Labelled instances TrAdaBoost")

df_pivot = pd.pivot_table(df, index = "Labelled", columns = "Strategy", values = "Only Target")
df_pivot.plot(figsize = (10,10), ylabel ="MCC score on Test dataset", xlabel = "Number of labelled instances", title = "Use only Labelled Target")

df_pivot = pd.pivot_table(df, index = "Labelled", columns = "Strategy", values = "Target and Source")
df_pivot.plot(figsize = (10,10), ylabel ="MCC score on Test dataset", xlabel = "Number of labelled instances", title = "Use all Labelled instances")

df_pivot = pd.pivot_table(df, index = "Labelled", columns = "Strategy", values = "Target Souce and TradaBoost")
df_pivot.plot(figsize = (10,10), ylabel ="MCC score on Test dataset", xlabel = "Number of labelled instances", title = "Use all Labelled instances TrAdaBoost")





