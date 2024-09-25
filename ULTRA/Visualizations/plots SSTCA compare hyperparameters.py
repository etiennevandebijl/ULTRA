from project_paths import get_results_df

df = get_results_df("test SSTCA balanced")
df = df[df["Test Data"] == "Eval"]
df = df[df["Weighting"] == False]


df = df.drop(['Feature Extractor',
              'Version', 
              'Protocol', 
              'Sizes subsets', 
              'experiment-name', 
              'Test Data', 
              "Weighting",
              "Size L_s", 
              "Model"], axis = 1)

df = df.fillna("NONE")

df = df.groupby(['Source Experiment', 'Target Experiment', 'Evaluation model',
                  'Size L_d', 'Size U', 'Number of components', 'Neighbours', 'Sigma', 'Lambda', 'Gamma','Mu',
                  'Train Set', 'Projection'])[["TP", "TN", "FP", "FN", "MCC"]].mean().reset_index()

df = df[(df["Sigma"] == "NONE") | (df["Projection"] == True)]


