from project_paths import get_results_df

experiment_name = "test AL strategies"
df = get_results_df(experiment_name)

df.head()

df = df.drop(["Unnamed: 0.1", "Unnamed: 0"], axis = 1)

df = df.rename(columns={'strategy': 'Strategy'})

df["Strategy"].value_counts()

cert_old = df["Strategy"] == "Certainty"
uncert_old = df["Strategy"] == "Uncertainty"

df.loc[cert_old, "Strategy"] = "Uncertainty"
df.loc[uncert_old, "Strategy"] = "Certainty"

df["Strategy"].value_counts()

df.to_csv("/home/etienne/Dropbox/Projects/ULTRA/Results/Tables/" +experiment_name +".csv", index = False)