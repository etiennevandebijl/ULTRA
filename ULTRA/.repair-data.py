import pandas as pd

from project_paths import get_results_df

# Revrieve results
df = get_results_df("active learning balanced dataset 2")
df.shape
df_ = df

df_["F_1"] = df_["F1"]

df_.drop(["F1"], axis = 1, inplace = True)

df_.columns = ['source_dataset','target_dataset', 'feature_extractor',
               'version',
               'protocol',
               'uniform_sample_size',
               'random_state_subset',
               'experiment_name',
               'model_eval',
               'random_state_eval',
               'query_size',
               'num_iterations',
               'uniform_tl_sample_size',
               'al_strategy',
               'train_al_with_weights',
               'update_projection',
               'update_weights',
               'l_s_size',
               'l_d_size',
               'u_size',
               'current_iteration',
               'training_set',
               'train_eval_with_weights',
                'train_eval_with_projection',
                'test_set',
                'tp',
                'tn',
                'fp',
                'fn',
                'recall',
                'prec',
                'acc',
                'f1',
                'mcc']

df_["model_ultra"] = df_['model_eval']

df_ = df_[['source_dataset','target_dataset', 'feature_extractor',
               'version',
               'protocol',
               'uniform_sample_size',
               'random_state_subset',
               'experiment_name',
               'model_ultra',
               'random_state_eval',
               'query_size',
               'num_iterations',
               'uniform_tl_sample_size',
               'al_strategy',
               'train_al_with_weights',
               'update_projection',
               'update_weights',
               'l_s_size',
               'l_d_size',
               'u_size',
               'current_iteration',
               'training_set',
               'model_eval',
               'train_eval_with_weights',
                'train_eval_with_projection',
                'test_set',
                'tp',
                'tn',
                'fp',
                'fn',
                'recall',
                'prec',
                'acc',
                'f1',
                'mcc']]

df_.to_csv("/home/etienne/Dropbox/Projects/ULTRA/Results/Tables/active learning balanced datasets 2.csv", index = False)

