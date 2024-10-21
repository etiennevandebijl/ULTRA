import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from project_paths import get_results_df

# %% Download data

df = get_results_df("test SSTCA-V3 target BM ratio 95 V4")
# Shape at 21 okt 2024 is 1924560, 43

# Remove not used variables
df = df.drop(['Feature Extractor',
              'Version', 
              'Protocol', 
              'Sizes subsets', 
              'experiment-name', 
              "Size L_s"], axis = 1)

# %%

EVAL_COLUMNS = ['TP', 'TN', 'FP', 'FN', 'Recall', 'Precision', 
                'Accuracy', 'F_1', 'MCC', 'ROC-AUC']
TCA_VARS = ['TCA Version', 'Number of components', 'Neighbours', 'Sigma', 
            'Lambda', 'Kernel', 'Gamma', 'Mu', 'semi_supervised', 
            'target_dependence', 'self_dependence']
TCA_OUTCOME_VARS = ['Objective score', 'Top eigenvalue', 'Sum eigenvalues']
RANDOM_VARS = ['Random_states subsets', 'Random state eval clf',
               'random state'] # Must be fixed
EXP_VARS = ['Source Experiment', 'Target Experiment', 'Size L_d', 'Size U', 'Evaluation model',
'Train Set',  'Test Data']

IGNORE = ["Model", 'Weighting', 'Projection']


# %%

df_ = df[df["Test Data"] == "Eval"]

df_ = df_.fillna("NONE")

# Take mean over the random seeds
df_ = df_.groupby(EXP_VARS + TCA_VARS)[EVAL_COLUMNS].mean().reset_index()

df_ = df_[df_["Train Set"] != "L_s"]

for combination, group in df_.groupby(['Source Experiment', 'Target Experiment', "Train Set" ]):
    plt.figure(figsize = (10,10))
    sns.boxplot(data = group,  x = "TCA Version", y = "MCC", hue = "Size L_d")
    plt.title(combination)
    plt.show()
       
       
