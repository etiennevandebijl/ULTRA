import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from project_paths import get_results_df

# %% Load data

df = get_results_df("test SSTCA-V2 target BM ratio 95")
print(df.shape) # (753060, 35)

# Only look at evaluation data
df = df[df["Test Data"] == "Eval"]

#No Weighting
df = df[df["Weighting"] == False]

# Not interested in L_S
df = df[df["Train Set"] != "L_s"]

# Exclude redudant variables
df = df.drop(['Feature Extractor', 'Version', 'Protocol', 'Sizes subsets', 
              'experiment-name', 'Test Data', "Weighting", "Model"], axis = 1)

# Variables SSTCA: 'Number of components', 'Neighbours', 'Sigma', 'Lambda', 'Gamma', 'Mu'
# Evaluation: Top eigenvalue, MCC

# RF does automatic feature-selection, so lets keep the original data space
# We can always look at reducing the dataspace, but results are not yet decisive.
df = df[df["Number of components"] == 8]
df.shape

# %%

for combination, group in df.groupby(["Source Experiment", "Target Experiment", "Size L_d"]):
    
    # Score without Projection
    group_B = group[(group["Projection"]==False)].groupby(["Train Set"])["MCC"].mean().to_dict()
    
    # Scores with Projection
    group_P = group[(group["Projection"]==True) & (group['Train Set'] == "L")]
    
    group_P = group_P.groupby(['Size L_d', 'Number of components', 'Neighbours', 
                                    'Sigma', 'Lambda', 'Gamma', 'Mu', 'Train Set', 
                                    'Projection'])[["MCC","Top eigenvalue"]].mean().reset_index()
        
    
    plt.figure(figsize = (10,10))
    
    # Plot baselines
    colours = {"L": "black", "L_d": "red"}
    for TS, mcc in group_B.items():
        plt.axhline(mcc, label = "MCC RF trained on " + str(TS), color = colours[TS])
        
        
        
    sns.scatterplot(x =np.log(np.abs(group_P["Top eigenvalue"])), y = group_P["MCC"], hue = group_P["Sigma"], palette='tab10')
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=6, fancybox=True)
    plt.tight_layout()
    plt.title("Source dataset: " + str(combination[0]) + " - Target dataset : " + str(combination[1]) + " - Number of labelled target instances " + str(combination[2]) )
    plt.xlabel("Log transformed Eigenvalue Component 1")
    plt.ylabel("MCC score Random Forest trained on L and tested on evaluation dataset")
    plt.show()    


#%% 

df = df[df["Lambda"]!=0.0]

for source_target, df_st in df.groupby(["Source Experiment", "Target Experiment"]):
    
    fig, axs = plt.subplots(1, 4, figsize = (25,8))
    i = 0
    
    min_MCC = df_st.groupby(['Size L_d', 'Number of components', 'Neighbours', 
                                'Sigma', 'Lambda', 'Gamma', 'Mu', 'Train Set', 
                                'Projection'])[["MCC"]].mean().min()[0]
    max_MCC = df_st.groupby(['Size L_d', 'Number of components', 'Neighbours', 
                                'Sigma', 'Lambda', 'Gamma', 'Mu', 'Train Set', 
                                'Projection'])[["MCC"]].mean().max()[0]
    
    for size, group in df_st.groupby(["Size L_d"]):

        group_B = group[(group["Projection"]==False)].groupby(["Train Set"])["MCC"].mean().to_dict()
        
        group_P = group[(group["Projection"]==True) & (group['Train Set'] == "L")]
        
        group_P = group_P.groupby(['Size L_d', 'Number of components', 'Neighbours', 
                                    'Sigma', 'Lambda', 'Gamma', 'Mu', 'Train Set', 
                                    'Projection'])[["MCC","Top eigenvalue"]].mean().reset_index()
        
        
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



df = df[df["Sigma"]=='1.0']

for source_target, df_st in df.groupby(["Source Experiment", "Target Experiment"]):
    
    fig, axs = plt.subplots(1, 4, figsize = (25,8))
    i = 0
    
    min_MCC = df_st.groupby(['Size L_d', 'Number of components', 'Neighbours', 
                                'Sigma', 'Lambda', 'Gamma', 'Mu', 'Train Set', 
                                'Projection'])[["MCC"]].mean().min()[0]
    max_MCC = df_st.groupby(['Size L_d', 'Number of components', 'Neighbours', 
                                'Sigma', 'Lambda', 'Gamma', 'Mu', 'Train Set', 
                                'Projection'])[["MCC"]].mean().max()[0]
    
    for size, group in df_st.groupby(["Size L_d"]):

        group_B = group[(group["Projection"]==False)].groupby(["Train Set"])["MCC"].mean().to_dict()
        
        group_P = group[(group["Projection"]==True) & (group['Train Set'] == "L")]
        
        group_P = group_P.groupby(['Size L_d', 'Number of components', 'Neighbours', 
                                    'Sigma', 'Lambda', 'Gamma', 'Mu', 'Train Set', 
                                    'Projection'])[["MCC","Top eigenvalue"]].mean().reset_index()
        
        
        colours = {"L": "black", "L_d": "red"}
        for TS, mcc in group_B.items():
            axs[i].axhline(mcc, label = "MCC RF trained on " + str(TS), color = colours[TS])
        
        sns.scatterplot(x =np.log(np.abs(group_P["Top eigenvalue"])), y = group_P["MCC"], hue = group_P["Neighbours"], palette='tab10',  ax=axs[ i])

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




# %% Analyse Lambda

for combination, group in df.groupby(["Source Experiment", "Target Experiment", "Size L_d", "Sigma"]):
    
    group_P = group[(group["Projection"]==True) & (group['Train Set'] == "L")]
    
    group_B = group[group["Number of components"].isnull()].groupby(["Train Set"])["MCC"].mean().to_dict()

    plt.figure(figsize = (10,10))
    colours = {"L": "black", "L_d": "red"}
    for TS, mcc in group_B.items():
        plt.axhline(mcc, label = "Score to Beat " + str(TS), color = colours[TS])
    sns.scatterplot(x =np.log(np.abs(group_P["Top eigenvalue"])), y = group_P["MCC"], hue = group_P["Neighbours"], palette='tab10')
    plt.title(combination)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=6, fancybox=True)
    plt.tight_layout()
    plt.xlabel("Log transformed Eigenvalue Component 1")
    plt.show()    



# %% 



# Weird high eigenvalues

df = df[df["Lambda"] > 0]
#df = df[df["Sigma"] != "1.0"]
df = df[df["Sigma"] == "MEAN"] # Leads to lower (maybe more realistic eigenvalues)

df = df[df["Size L_d"] == 100]




df_summary_1 = pd.pivot_table(df, index = ["Source Experiment", "Target Experiment", "Train Set"], columns = ["Lambda", "Neighbours", 'Size L_d'], values = "MCC").T


df_summary_2 = pd.pivot_table(df, index = ["Source Experiment", "Target Experiment", "Train Set"], columns = ["Lambda", "Neighbours", 'Size L_d'], values = "Top eigenvalue").T


#df = df[df["Top eigenvalue"] < 100000]




for combination, group in df.groupby(["Source Experiment", "Target Experiment", 'Train Set']):
    if combination[2] != "L":
        continue
    plt.figure(figsize = (10,10))
    sns.scatterplot(x = np.log(np.abs(group["Top eigenvalue"])), y = group["MCC"], hue = df["Sigma"], palette='tab10')
    plt.title(combination)
    plt.show()
    

for combination, group in df.groupby(["Source Experiment", "Target Experiment", 'Train Set']):
    if combination[2] != "L":
        continue
    plt.figure(figsize = (10,10))
    sns.boxplot(x =group["Lambda"], y = group["MCC"], hue = group["Size L_d"], palette='tab10')
    plt.title(combination)
    plt.show()    

    






# This is one to remember; median leads to higher eigenvalues compared to mean
for combination, group in df.groupby(["Source Experiment", "Target Experiment", 'Train Set']):
    plt.figure(figsize = (10,10))
    sns.boxplot(x = np.log(np.abs(group["Top eigenvalue"])), y = group["MCC"], hue = df["Lambda"])
    plt.title(combination)
    plt.show()
    
