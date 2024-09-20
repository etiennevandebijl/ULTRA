import numpy as np

from project_paths import get_results_df

experiment_name = "active-learning-v4"
df = get_results_df(experiment_name)


df["Recall"] = df["TP"] / (df["TP"] + df["FN"])
df["Precision"] = df["TP"] / (df["TP"] + df["FP"])
df["Accuracy"] = (df["TP"] + df["TN"]) / (df["TP"] + df["TN"] + df["FN"] + df["FP"])
df["F1"] = (2 * df["TP"]) / (2 * df["TP"] + df["FN"] + df["FP"])
df["MCC"] = (df["TP"] * df["TN"] - df["FP"] * df["FN"]) / np.sqrt((df["TP"] + df["FP"]) * (df["TP"] + df["FN"]) * (df["TN"] + df["FP"]) * (df["TN"] + df["FN"]))


df.to_csv("/home/etienne/Dropbox/Projects/ULTRA/Results/Tables/" +experiment_name +".csv")