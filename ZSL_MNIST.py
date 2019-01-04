# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 19:50:50 2018

@author: Christopher
"""

import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from ZSL_models import RelationNetwork, AttributeNetwork
from load_data import load_MNIST_ZSL
from utils2 import confusion_matrix, batch


# atributes CxA matrix with attributes: (1) has corners, (2) has vertical lines, (3) has horizontal lines, (4) has circle, (5) has curves
attributes = [[0,0.5,0.5,1,1],[1,1,0,0,0],[1,0,1,0,1],[0,0.5,0.8,0,1],[1,1,1,0,0],[1,1,1,0,1],[0,0.5,0.5,1,1],[1,0.8,1,0,0],[0,0.5,0.5,1,1], [0,0.5,0,1,1]]
attributes = np.asarray([np.asarray(row) for row in attributes])
attributes = np.reshape(attributes, newshape=(1,-1))
print(attributes.shape)
#attributes = np.zeros(shape=(10,5))
#import matplotlib.pyplot as plt
#plt.imshow(digits['images'][104])
#print(digits['target'][104])


#(train, train_labels, OH_train_labels), (test, test_labels, OH_test_labels) = load_MNIST()
(train, train_labels, OH_train_labels), (test, test_labels, OH_test_labels) = load_MNIST_ZSL(without = [5,6,7,8,9])


# number of neurons in each layer
input_num_units = 8*8
hidden_num_units = 100
output_num_units = 10

# set remaining variables
epochs = 30
batch_size = 128
learning_rate = 0.0001
nr_batches = 479

# define model
rel_model = RelationNetwork(100, hidden_num_units, output_num_units)
att_model = AttributeNetwork(input_num_units, hidden_num_units, 50)
loss_fn = torch.nn.MSELoss()

# define optimization algorithm
rel_optimizer = torch.optim.Adam(rel_model.parameters(), lr=learning_rate)
att_optimizer = torch.optim.Adam(att_model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    train_batch = np.split(train,nr_batches)
    label_batch = np.split(train_labels,nr_batches)
    OH_batch = np.split(OH_train_labels,nr_batches)
    for i in range(nr_batches):
        x, y = Variable(torch.from_numpy(train_batch[i]).float()), Variable(torch.from_numpy(OH_batch[i]).float(), requires_grad=False)
        # pass that batch trough attribute network
        pred = att_model(x)
        # concat each row with its respective attributes from 'attributes'
        att_matrix = np.asarray([attributes for j in label_batch[i]]).reshape(len(train_batch[i]), 50)
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
        
    print(epoch, loss.data)
        
    if epoch%100 == 0:
        print(epoch, "loss", loss.data)

        # testing:
        
        x, y = Variable(torch.from_numpy(train).float()), Variable(torch.from_numpy(OH_train_labels).float(), requires_grad=False)
        pred = att_model(x)
        att_matrix = np.asarray([attributes for j in train_labels]).reshape(len(train), 50)
        att_matrix = Variable(torch.from_numpy(att_matrix).float())
        combined = torch.cat((pred, att_matrix), 1)

        pred = rel_model(combined)
        # get loss
        print(loss_fn(pred, y))
        final_pred = np.argmax(pred.data.numpy(), axis=1)
        final_pred = [f+1 for f in final_pred]
        #print(final_pred, train_labels)
        
        print("acc on train set", accuracy_score(train_labels, final_pred))


x, y = Variable(torch.from_numpy(test).float()), Variable(torch.from_numpy(OH_test_labels).float(), requires_grad=False)
pred = att_model(x)
att_matrix = np.asarray([attributes for j in test_labels]).reshape(len(test), 50)
att_matrix = Variable(torch.from_numpy(att_matrix).float())
combined = torch.cat((pred, att_matrix), 1)

pred = rel_model(combined)

print(loss_fn(pred, y))
final_pred = np.argmax(pred.data.numpy(), axis=1)
final_pred = [f+1 for f in final_pred]
print("acc on test set", accuracy_score(test_labels, final_pred))

confusion_matrix(test_labels, final_pred, size=10)































