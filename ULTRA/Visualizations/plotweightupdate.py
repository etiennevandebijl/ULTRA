import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


def plot_weight_update(w_list, L_s, L_d):
    
    source_average = []
    target_average = []
    source_w_t_list = []
    target_w_t_list = []
    
    for w_t in w_list:
        w_t_source = w_t[L_s]
        w_t_target = w_t[L_d]
        source_w_t_list.append(w_t_source)
        target_w_t_list.append(w_t_target)
        source_average.append(np.mean(w_t_source))
        target_average.append(np.mean(w_t_target))
        
    fig, ax = plt.subplots()
    ax.boxplot([source_w_t_list[-1], target_w_t_list[-1]], positions=[0, 1])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Source', 'Target'])
    plt.ylabel("Weight v")
    plt.show()
    
    
    plt.figure(figsize = (15,8))
    sns.heatmap(source_w_t_list)
    plt.title("Progress weight vector v in Source instances")
    plt.xlabel("Source Instance")
    plt.ylabel("Weight V")
    plt.show()
    
    plt.figure(figsize = (15,8))
    sns.heatmap(target_w_t_list)
    plt.title("Progress weight vector v in Target instances")
    plt.xlabel("Target Instance")
    plt.ylabel("Weight V")
    plt.show()