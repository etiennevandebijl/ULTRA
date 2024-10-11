import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from project_paths import get_results_df

# %% Load data

df = get_results_df("test SSTCA-V2 target BM ratio 95 V2")
print(df.shape) # (753060, 35)

# Only look at evaluation data
df = df[df["Test Data"] == "Eval"]

#No Weighting
df = df[df["Weighting"] == False]

# Not interested in L_S
df = df[df["Train Set"] != "L_s"]

# Exclude redudant variables

# Select RF
df = df[df["Model"] == "RF"]

df = df.drop(['Feature Extractor', 'Version', 'Protocol', 'Sizes subsets', 
              'experiment-name', 'Test Data', "Weighting", "Model"], axis = 1)

# Variables SSTCA: 'Number of components', 'Neighbours', 'Sigma', 'Lambda', 'Gamma', 'Mu'
# Evaluation: Top eigenvalue, MCC

# RF does automatic feature-selection, so lets keep the original data space
# We can always look at reducing the dataspace, but results are not yet decisive.
df = df[df["Number of components"] == 8]
df.shape


#%% 
cols = ['Size L_d', 'Number of components', 'Neighbours', 'Sigma', 'Lambda', 
        'Gamma', 'Mu', 'Train Set', 'Projection']

#df = df[df["Sigma"] =='1.0']
#df = df[df["Lambda"] == 1.0]
#df = df[df["Neighbours"] == 100]

for source_target, df_st in df.groupby(["Source Experiment", "Target Experiment"]):
    
    fig, axs = plt.subplots(1, 4, figsize = (25,8))
    i = 0
    
    min_MCC = df_st.groupby(cols)[["MCC"]].mean().min()[0]
    max_MCC = df_st.groupby(cols)[["MCC"]].mean().max()[0]
    
    for size, group in df_st.groupby(["Size L_d"]):

        group_B = group[(group["Projection"]==False)].groupby(["Train Set"])["MCC"].mean().to_dict()
        
        group_P = group[(group["Projection"]==True) & (group['Train Set'] == "L")]
        
        group_P = group_P.groupby(cols)[["MCC", "Top eigenvalue", "Sum eigenvalues"]].mean().reset_index()
        
        
        colours = {"L": "black", "L_d": "red"}
        for TS, mcc in group_B.items():
            axs[i].axhline(mcc, label = "MCC RF trained on " + str(TS), color = colours[TS])
        
        sns.scatterplot(x =np.log(np.abs(group_P["Top eigenvalue"])), y = group_P["MCC"], hue = group_P["Sigma"], palette='tab10',  ax=axs[ i])

        axs[i].set_title("Number of labelled target instances: " + str(size[0]) )
        axs[i].set_xlabel("Log transformed Eigenvalue Component 1")
        axs[i].set_ylabel("MCC score Random Forest trained on L and tested on evaluation dataset")
        handles, labels = axs[i].get_legend_handles_labels()
        axs[i].get_legend().remove()
        axs[i].set_ylim(min_MCC,max_MCC)
        i = i + 1
    fig.legend(handles, labels, loc='upper right', ncol = 5)
    plt.suptitle("Source dataset: " + str(source_target[0]) + " - Target dataset : " + str(source_target[1]))
    plt.show()    

# %%

for source_target, df_st in df.groupby(["Source Experiment", "Target Experiment"]):
    

    
    fig, axs = plt.subplots(1, 4, figsize = (25,8))
    i = 0
    

    for size, group in df_st.groupby(["Size L_d"]):

        group_B = group[(group["Projection"]==False)].groupby(["Train Set"])["MCC"].mean().to_dict()
        
        group_P = group[(group["Projection"]==True) & (group['Train Set'] == "L")]


        sns.boxplot(x =group_P["Lambda"], y = group_P["MCC"], hue = group_P["Neighbours"], palette='tab10',  ax=axs[ i])
        handles, labels = axs[i].get_legend_handles_labels()
        axs[i].get_legend().remove()
        axs[i].set_ylim(min_MCC,max_MCC)
        i = i + 1
    fig.legend(handles, labels, loc='upper right', ncol = 5)
    plt.suptitle("Source dataset: " + str(source_target[0]) + " - Target dataset : " + str(source_target[1]))
    plt.show()  




