# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 12:28:37 2018

@author: Christopher
"""

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
from load_data import load_hyp_spectral_splitted, load_attributes
from utils2 import confusion_matrix
#from utils import load_data



#(train, train_labels, OH_train_labels), (test, test_labels, OH_test_labels) = load_MNIST()
(train, train_labels, OH_train_labels), (test, test_labels, OH_test_labels) = load_hyp_spectral_splitted(without = [0])
#(train, train_labels), (test, test_labels) = load_data()
#print(train.shape, train_labels.shape, test.shape, test_labels.shape)

# number of neurons in each layer
input_num_units = 220
hidden_num_units = 1000
output_num_units = 16

# set remaining variables
epochs = 20
batch_size = 128
learning_rate = 0.001
#'hyp_simple_features.txt', 
att_data = ['abstract_features_idea1.txt','abstract_features_idea2.txt', 'abstract_features_idea3.txt']
att_data = ['abstract_features_idea1_selected']

trials = 1
results = []

for file in att_data[0:1]:
    result_row = []
    print(file)
    #load attributes:
    attributes = load_attributes(file)
#    for key in attributes:
#        print(key, len(attributes[key]))
    for trial in range(trials):
        print(trial)
        # define model
        rel_model = RelationNetwork(len(attributes[1])*2, hidden_num_units, output_num_units)
        att_model = AttributeNetwork(input_num_units, hidden_num_units, len(attributes[1]))
        loss_fn = torch.nn.MSELoss() # TODO: maybe set weights for the classes to balance the loss.
        
        # define optimization algorithm
        rel_optimizer = torch.optim.Adam(rel_model.parameters(), lr=learning_rate, weight_decay=1e-6)
        att_optimizer = torch.optim.Adam(att_model.parameters(), lr=learning_rate, weight_decay=1e-5)
       
        
        print("train")
        for epoch in range(epochs):
            #print(torch.from_numpy(train.astype(float)).float())
            x, y = Variable(torch.from_numpy(train.astype(float)).float()), Variable(torch.from_numpy(OH_train_labels).float(), requires_grad=False)
            # pass that batch trough attribute network
            pred = att_model(x)
            # concat each row with its respective attributes from 'attributes'
#            att_matrix = np.zeros(shape=(16, len(attributes[1])))
#            for i in range(1,17):
#                att_matrix[i-1] = attributes[i]
#            att_matrix = np.asarray(att_matrix)
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
            #print(pred[50:55], y[50:55])
        
        
        # testing:
        print("test")
        x, y = Variable(torch.from_numpy(train.astype(float)).float()), Variable(torch.from_numpy(OH_train_labels).float(), requires_grad=False)
        pred = att_model(x)
        att_matrix = np.asarray([attributes[i] for i in train_labels])
        att_matrix = Variable(torch.from_numpy(att_matrix).float())
        combined = torch.cat((pred, att_matrix), 1)
        
        pred = rel_model(combined)
        # get loss
        print(loss_fn(pred, y))
        
        final_pred = np.argmax(pred.data.numpy(), axis=1)
        
        print("acc on train set", accuracy_score(train_labels, final_pred))
        
        
        x, y = Variable(torch.from_numpy(test.astype(float)).float()), Variable(torch.from_numpy(OH_test_labels).float(), requires_grad=False)
        pred = att_model(x)
        att_matrix = np.asarray([attributes[i] for i in test_labels])
        att_matrix = Variable(torch.from_numpy(att_matrix).float())
        combined = torch.cat((pred, att_matrix), 1)
        
        pred = rel_model(combined)
        print(loss_fn(pred, y))
        final_pred = np.argmax(pred.data.numpy(), axis=1)
        acc = accuracy_score(test_labels, final_pred)
        print("acc on test set", acc)
        confusion_matrix(final_pred, test_labels)
        result_row.append(acc)
        
    result_row.append(np.average(result_row))
    results.append(result_row)

for r in results:
    print(r)
    





