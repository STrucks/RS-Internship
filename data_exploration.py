# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 09:28:50 2018

@author: Christopher
"""

from load_data import load_hyp_spectral, load_attributes
import numpy as np
import matplotlib.pyplot as plt

data = load_hyp_spectral()

def plot_avg():
    # lets first compare the average complete bandwidth of every material:
    avg_material = np.asarray([np.average(data[str(i)], axis = 0) for i in range(1,17)])
    #print(avg_material)
    
    import seaborn as sns
    #plt.imshow(avg_material, cmap='hot', interpolation='nearest', linew)
    print(avg_material[0:2,0:10])
    plt.figure(figsize=(40,5))
    ax = sns.heatmap(avg_material, linewidth=1)
    #plt.show()

def show_data_distr():
    # show data distribution:
    data_size = np.sum([len(data[x]) for x in data])
    for key in data:
        print(key, len(data[key])/data_size)
        
def show_attribute_matrix():
    file = "abstract_features_idea1_selected"
    att = load_attributes(file)
    import seaborn as sns
    #plt.imshow(avg_material, cmap='hot', interpolation='nearest', linew)
    att_matrix = np.asarray([att[i][0:50] for i in range(1,17)])
    print(att_matrix.shape)
    plt.figure(figsize=(20,5))
    ax = sns.heatmap(att_matrix, linewidth=1)
    
show_attribute_matrix()