import seaborn as sns
import matplotlib.pyplot as plt

from project_paths import get_results_df

# %% Download data

df = get_results_df("test SSTCA-V3 target BM ratio 95 V4")
# Shape at 21 okt 2024 is 1924560, 43

# Remove not used variables
df = df.drop(['feature_extractor',
              'version', 
              'protocol', 
              'uniform_sample_size', 
              'experiment_name', 
              "l_s_size"], axis = 1)

# %%

EVAL_COLUMNS = ['tp', 'tn', 'fp', 'fn', 'recall', 'prec', 
                'acc', 'f1', 'mcc', 'roc_auc']

TCA_VARS = ['tca_variant', 'num_components', 'num_neighbours', 'sigma', 
            'lambda', 'kernel', 'gamma', 'mu', 'semi_supervised', 
            'target_dependence', 'self_dependence']

TCA_OUTCOME_VARS = ['objective_value', 'highest_abs_eigenvalue', 'sum_abs_eigenvalues']

RANDOM_VARS = ['random_state_subset', 'random_state_eval', 'random_state_tca'] 

EXP_VARS = ['source_dataset', 'target_dataset', 'l_d_size', 'u_size',
            'model_eval', 'training_set',  'test_set']

IGNORE = ['train_eval_with_weights', 'train_eval_with_projection']


# %%

df_ = df[df["test_set"] == "Eval"]

df_ = df_.fillna("NONE")

# Take mean over the random seeds
df_ = df_.groupby(EXP_VARS + TCA_VARS)[EVAL_COLUMNS].mean().reset_index()

df_ = df_[df_["training_set"] != "L_s"]

for combination, group in df_.groupby(['source_dataset', 'target_dataset', "training_set"]):
    plt.figure(figsize = (10,10))
    sns.boxplot(data = group,  x = "tca_variant", y = "mcc", hue = "l_d_size")
    plt.title(combination)
    plt.show()
       
       
