# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 10:05:31 2018

@author: Christopher
"""

import numpy as np
from matplotlib import pyplot as plt

def confusion_matrix(y, t, size = 16):
    print("Confusion Matrix")
    #print(y,t)
    
    matrix = np.zeros(shape=(size, size))
    for ys, ts in zip(y,t):
        matrix[int(ys)-1,int(ts)-1] += 1
    print(matrix)
    
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def heatmap(matrix, title = "heatmap"):
    import seaborn as sns
    
    plt.figure(figsize=(1*len(matrix),1*len(matrix[0])))
    #plt.subplot(4,1,1)
    plt.title(title)
    ax1 = sns.heatmap(matrix, linewidth=1)
    