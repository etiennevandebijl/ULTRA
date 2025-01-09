import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from project_paths import PROJECT_PATH, go_or_create_folder

import warnings
warnings.filterwarnings('ignore')

IGNORE = ['feature_extractor', 'version', 'protocol', 'uniform_sample_size', 
          'experiment_name',  "l_s_size", "train_eval_with_projection", 
          "train_eval_with_weights"]

EXP_VARS = ['source_dataset', 'target_dataset', 'l_d_size', 'u_size',
            'model_eval', 'training_set',  'test_set']

TCA_VARS = ['tca_variant', 'num_components', 'mu', 'kernel', 'semi_supervised']

SSTCA_VARS = ['num_neighbours', 'sigma', 'lambda', 'gamma', 
              'target_dependence', 'self_dependence']

RANDOM_VARS = ['random_state_subset', 'random_state_eval', 'random_state_tca'] 

EVAL_COLUMNS = ['tp', 'tn', 'fp', 'fn', 'recall', 'prec', 
                'acc', 'f1', 'mcc', 'roc_auc']

TCA_OUTCOME_VARS = ['objective_value', 'highest_abs_eigenvalue', 'sum_abs_eigenvalues']


def plot_source_target(data, hue, hue_title, extra_info = "",
                        x_var = "model_eval", x_label = "Classification model",
                        y_var = "mcc", y_label = "MCC score evaluation (target) dataset",
                        loc_legend = (0.72, 0.92),  
                        subtitle = "", plot_type = "boxplot", 
                        experiment = "Experiment TCA compare hyperparameters target BM ratio 95 V1/",
                        plot_num_obs = False
                        ):
    
    save_path = PROJECT_PATH + "Results/Figures/" + experiment
    
    plot_name = plot_type + "-" + x_var + " vs " + y_var + "-hue " + hue + "--" + extra_info
    plot_name = plot_name.replace("_", " ")
    go_or_create_folder(save_path, plot_name)
    
    y_min = max(data[y_var].min() / 1.05, -1)
    y_max = min(data[y_var].max() * 1.05, 1)
    
    for source_target, df_st in data.groupby(["source_dataset", "target_dataset"]):
    
        fig, axs = plt.subplots(1, 4, figsize = (19,7))
        i = 0
    
        for size, group in df_st.groupby(["l_d_size"]):
            
            if plot_type == "boxplot":
                
                x_var_unique = sorted(list(group[x_var].unique()), key=lambda x: (isinstance(x, str), x))
                hue_unique = sorted(list(group[hue].unique()), key=lambda x: (isinstance(x, str), x))
                
                sns.boxplot(data = group, x = x_var, y = y_var, hue = hue, 
                            order = x_var_unique,
                            hue_order = hue_unique, ax=axs[i], palette='tab10')
                
                # Plot number of observations
                
                if plot_num_obs:
                    group_data = group.groupby([x_var, hue]).agg({y_var:["median", "count"]}).reset_index()
                    group_data[x_var] = pd.Categorical(group_data[x_var]).codes
                    group_data[hue] = pd.Categorical(group_data[hue]).codes
                    
                    width = 0.8  # Default box width
                    hue_offset = width / len(hue_unique)  # Space for each hue
                    base_offset = -width / 2 + hue_offset / 2  # Starting position
                    
                    group_data["x_position"] = group_data[x_var] + base_offset + group_data[hue] * hue_offset
                    
                    for _, row in group_data.iterrows():
                        axs[i].text(row["x_position"], row[y_var]["median"], str(int(row[y_var]['count'])),
                                ha='center', va='bottom', fontsize=10, color='black')
                

            if plot_type == "scatterplot":
                sns.scatterplot(data = group, x = x_var, y = y_var, hue = hue, ax=axs[i], palette='tab10')
            
            axs[i].set_title("Number of labeled target instances: " + str(size[0]))
            
            axs[i].set_xlabel(x_label)
            axs[i].set_ylabel(y_label)
            handles, labels = axs[i].get_legend_handles_labels()
            axs[i].get_legend().remove()
            
            axs[i].set_ylim(y_min, y_max)
            if i > 0:
                axs[i].set_ylabel('')
                
            i = i + 1
        
        fig.legend(handles, labels, loc=loc_legend, title = hue_title, ncol = len(data[hue].unique()))
        fig.text(0.5, 0.93, subtitle,  ha="center")
        plt.suptitle("Source dataset: " + str(source_target[0]) + " - Target dataset : " + str(source_target[1]))
        plt.savefig(save_path + plot_name + "/" + plot_name +" plot " +" ".join(source_target) + ".png", bbox_inches='tight')
        plt.show()  
