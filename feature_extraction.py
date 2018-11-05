# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:32:12 2018

@author: Christopher
"""

import numpy as np

def extract_simple_features():
    """
    Before we use more complicated methods, like pretrained NN, lets start with simple features:
    - simply the average over 50 channels (400-405, 406-410 ect)
    """
    # load the data
    from scipy.io import loadmat
    
    f = loadmat("data/indian_pines.mat")
    raw_data = f['indian_pines']
    
    f = loadmat("data/indian_pines_gt.mat")
    GT = f['indian_pines_gt']
    
    data = {}
    
    for row in range(len(GT)):
        for col in range(len(GT[row,:])):
            if str(GT[row,col]) in data:
                data[str(GT[row,col])].append(raw_data[row, col, :])
            else:
                data[str(GT[row, col])] = []
                data[str(GT[row, col])].append(raw_data[row, col, :])
    
    features = []
    for key in data.keys():
        averages = sum(data[key])
        averages = np.dot(averages, 1/len(data[key]))
        channels = np.split(averages, 20)
        row = [sum(c)/len(c) for c in channels]
        features.append([key] + row)
        
    print(features)
    with open("hyp_simple_features.txt", 'w') as f:
        for row in features:
            f.write(row[0] + ":")
            for nr in row[1:-1]:
                f.write(str(nr) + ",")
            f.write(str(row[-1]) + "\n")

extract_simple_features()
    
    
    