# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 09:28:50 2018

@author: Christopher
"""

from load_data import load_hyp_spectral, load_attributes, load_hyp_spectral_preprocessed
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
    distr = {}
    for key in sorted(data):
        print(key, len(data[key])/data_size)
        distr[int(key)] = len(data[key])/data_size
    distr = [distr[k] for k in list(sorted(distr))]
    plt.subplot(2,1,1)
    plt.bar(range(1,17), distr)
    plt.title("data distribution before preprocessing")
    data_p = load_hyp_spectral_preprocessed()
    distr = {}
    data_size = np.sum([len(data_p[x]) for x in data_p])
    for key in data_p:
        print("--",key, len(data_p[key])/data_size)
        distr[int(key)] = len(data_p[key])/data_size
    distr = [distr[k] for k in list(sorted(distr))]
    plt.subplot(2,1,2)
    plt.bar(range(1,17), distr)
    plt.title("data distribution after preprocessing")
    plt.show()

        
def show_attribute_matrix():
    import seaborn as sns
    
    plt.figure(figsize=(10,20))
    plt.subplot(4,1,1)
    plt.title("idea 1")
    file = "abstract_features_idea1_selected"
    att = load_attributes(file)
    att_matrix = np.asarray([att[str(i)][0:10] for i in range(1,17)])
    print(att_matrix.shape)
    ax1 = sns.heatmap(att_matrix, linewidth=1)
    
    plt.subplot(4,1,2)
    plt.title("idea 2")
    file = "abstract_features_idea2_selected"
    att = load_attributes(file)
    att_matrix = np.asarray([att[str(i)][0:10] for i in range(1,17)])
    ax2 = sns.heatmap(att_matrix, linewidth=1)
    
    """
    plt.subplot(5,1,3)
    plt.title("idea 3")
    file = "abstract_features_idea3_selected"
    att = load_attributes(file)
    att_matrix = np.asarray([att[i][0:50] for i in range(1,17)])
    ax3 = sns.heatmap(att_matrix, linewidth=1)
    """
    plt.subplot(4,1,3)
    plt.title("autoencoder_features_10_e50")
    file = "autoencoder_features_10_e50"
    att = load_attributes(file)
    att_matrix = np.asarray([att[str(i)] for i in range(1,17)])
    ax4 = sns.heatmap(att_matrix, linewidth=1)
    
    plt.subplot(4,1,4)
    plt.title("VAE_features_10_e50")
    file = "VAE_features_10_e50"
    att = load_attributes(file)
    att_matrix = np.asarray([att[str(i)] for i in range(1,17)])
    ax4 = sns.heatmap(att_matrix, linewidth=1)
    
    
    
       
    
show_attribute_matrix()