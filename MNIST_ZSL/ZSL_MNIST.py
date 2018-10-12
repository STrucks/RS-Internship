# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 19:50:50 2018

@author: Christopher
"""

import torch
from sklearn.datasets import load_digits
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from ZSL_models import RelationNetwork, AttributeNetwork

def one_hot(labels, nr_classes):
    _labels = []
    for l in labels:
        _labels.append([0]*nr_classes)
        _labels[-1][l] = 1
    return np.asarray(_labels)


digits = load_digits()
# atributes CxA matrix with attributes: (1) has corners, (2) has vertical lines, (3) has horizontal lines, (4) has circle, (5) has curves
attributes = [[0,0.5,0.5,1,1],[1,1,0,0,0],[1,0,1,0,1],[0,0.5,0.8,0,1],[1,1,1,0,0],[1,1,1,0,1],[0,0.5,0.5,1,1],[1,0.8,1,0,0],[0,0.5,0.5,1,1], [0,0.5,0,1,1]]
attributes = np.asarray([np.asarray(row) for row in attributes])
#import matplotlib.pyplot as plt
#plt.imshow(digits['images'][104])
#print(digits['target'][104])

data = [np.reshape(img,newshape=(1,8*8)) for img in digits['images']]
data = [d[0] for d in data]
labels = digits['target']

train, test = np.asarray(data[0:int(0.8*len(data))]), np.asarray(data[int(0.8*len(data)):])
train_labels , test_labels = np.asarray(labels[0:int(0.8*len(labels))]), np.asarray(labels[int(0.8*len(labels)):])
OH_train_labels, OH_test_labels = one_hot(np.asarray(labels[0:int(0.8*len(labels))]), 10), one_hot(np.asarray(labels[int(0.8*len(labels)):]), 10)

# number of neurons in each layer
input_num_units = 8*8
hidden_num_units = 1000
output_num_units = 10

# set remaining variables
epochs = 500
batch_size = 128
learning_rate = 0.0001






# define model
rel_model = RelationNetwork(10, hidden_num_units, output_num_units)
att_model = AttributeNetwork(input_num_units, hidden_num_units, 5)
loss_fn = torch.nn.MSELoss()

# define optimization algorithm
rel_optimizer = torch.optim.Adam(rel_model.parameters(), lr=learning_rate)
att_optimizer = torch.optim.Adam(att_model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    x, y = Variable(torch.from_numpy(train).float()), Variable(torch.from_numpy(OH_train_labels).float(), requires_grad=False)
    # pass that batch trough attribute network
    pred = att_model(x)
    # concat each row with its respective attributes from 'attributes'
    att_matrix = np.asarray([attributes[i] for i in train_labels])
    att_matrix = Variable(torch.from_numpy(att_matrix).float())
    combined = torch.cat((pred, att_matrix), 1)
    
    pred = rel_model(combined)
    # get loss
    loss = loss_fn(pred, y)

    # perform backpropagation
    att_model.zero_grad()
    rel_model.zero_grad()
    loss.backward()
    att_optimizer.step()
    rel_optimizer.step()
    
    
    print(epoch, "loss", loss.data)



# testing:

x, y = Variable(torch.from_numpy(train).float()), Variable(torch.from_numpy(OH_train_labels).float(), requires_grad=False)
pred = att_model(x)
att_matrix = np.asarray([attributes[i] for i in train_labels])
att_matrix = Variable(torch.from_numpy(att_matrix).float())
combined = torch.cat((pred, att_matrix), 1)

pred = rel_model(combined)
# get loss
print(loss_fn(pred, y))

final_pred = np.argmax(pred.data.numpy(), axis=1)

print("acc on train set", accuracy_score(train_labels, final_pred))


x, y = Variable(torch.from_numpy(test).float()), Variable(torch.from_numpy(OH_test_labels).float(), requires_grad=False)
pred = att_model(x)
att_matrix = np.asarray([attributes[i] for i in test_labels])
att_matrix = Variable(torch.from_numpy(att_matrix).float())
combined = torch.cat((pred, att_matrix), 1)

pred = rel_model(combined)

print(loss_fn(pred, y))
final_pred = np.argmax(pred.data.numpy(), axis=1)
print("acc on test set", accuracy_score(test_labels, final_pred))






