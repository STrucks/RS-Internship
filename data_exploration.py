# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 09:28:50 2018

@author: Christopher
"""

from load_data import load_hyp_spectral
import numpy as np
import matplotlib.pyplot as plt

data = load_hyp_spectral()

# lets first compare the average complete bandwidth of every material:
avg_material = np.asarray([np.average(data[str(i)], axis = 0) for i in range(17)])
#print(avg_material)

import seaborn as sns
#plt.imshow(avg_material, cmap='hot', interpolation='nearest', linew)
print(avg_material[0:2,0:10])
plt.figure(figsize=(40,5))
ax = sns.heatmap(avg_material, linewidth=1)
#plt.show()
