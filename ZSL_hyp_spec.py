# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 12:28:37 2018

@author: Christopher
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 19:50:50 2018

@author: Christopher

with MH preprocessed:
[]      | 74%, 0.15
[16]    |

"""

import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from ZSL_models import RelationNetwork, AttributeNetwork
from load_data import load_hyp_spectral_splitted, load_attributes, load_hyp_spectral_splitted_preprocessed, preprocess, one_hot
from utils2 import confusion_matrix
from utils import load_data
#from utils import load_data



#(train, train_labels, OH_train_labels), (test, test_labels, OH_test_labels) = load_MNIST()
#(train, train_labels, OH_train_labels), (test, test_labels, OH_test_labels) = preprocess(exclude = [0])

(train, train_labels, OH_train_labels), (test, test_labels, OH_test_labels) = load_hyp_spectral_splitted()
#train, train_labels, test, test_labels, _ = load_data(root_dir='./',dataset_name='PINE', scale_dataset=True, perc_split = [80,20,0])
#OH_train_labels = one_hot(train_labels,16)
#OH_test_labels = one_hot(test_labels,16)

#print(train.shape, train_labels.shape, test.shape, test_labels.shape)
print(OH_train_labels)
print(min(train_labels),min(test_labels))
# number of neurons in each layer
input_num_units = 220
hidden_num_units = 100
output_num_units = 16
# set remaining variables 
epochs = 20
batch_size = 128
learning_rate = 0.001
nr_batches = len(train)#8194#241
nr_attributes = 10 * 16
#'hyp_simple_features.txt', 
att_data = ['abstract_features_idea1.txt','abstract_features_idea2.txt', 'abstract_features_idea3.txt']
att_data = ['autoencoder_features_10_e50']
# att_data = ['abstract_features_idea1']


trials = 3
results = []

for file in att_data[0:1]:
    result_row = []
    print(file)
    #load attributes:
    attributes = load_attributes(file, 10)
    attributes = np.asarray([np.asarray(attributes[str(key)]) for key in range(1,17)])
    attributes = np.reshape(attributes, newshape=(1,-1))
    print(attributes.shape)

    #attributes = np.eye(17)
#    for key in attributes:
#        print(key, len(attributes[key]))
    for trial in range(trials):
        print(trial)
        # define model
        rel_model = RelationNetwork(nr_attributes + 10, hidden_num_units, output_num_units)
        att_model = AttributeNetwork(input_num_units, hidden_num_units, 10)
        loss_fn = torch.nn.MSELoss() # TODO: maybe set weights for the classes to balance the loss.
        
        # define optimization algorithm
        rel_optimizer = torch.optim.Adam(rel_model.parameters(), lr=learning_rate, weight_decay=1e-6)
        att_optimizer = torch.optim.Adam(att_model.parameters(), lr=learning_rate, weight_decay=1e-5)
       
        #print(len(train))
        print("train")
        for epoch in range(epochs):# how many times we iterate over the whole dataset
            
            train_batch = np.split(train,nr_batches)
            label_batch = np.split(train_labels,nr_batches)
            OH_batch = np.split(OH_train_labels,nr_batches)
            for b in range(nr_batches):
                #print(torch.from_numpy(train.astype(float)).float())
                x, y = Variable(torch.from_numpy(train_batch[b].astype(float)).float()), Variable(torch.from_numpy(OH_batch[b]).float(), requires_grad=False)
                # pass that batch trough attribute network
                pred = att_model(x)
                # concat each row with the attribute matrix
                att_matrix = np.asarray([attributes for j in label_batch[b]]).reshape(len(train_batch[b]), nr_attributes)
                att_matrix = Variable(torch.from_numpy(att_matrix).float())
                combined = torch.cat((pred, att_matrix), 1)
                
                pred = rel_model(combined)
                #print(pred[0], y[0])
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
        
            if epoch%10 == 0:
                # testing:
                print("test")
                x, y = Variable(torch.from_numpy(train.astype(float)).float()), Variable(torch.from_numpy(OH_train_labels).float(), requires_grad=False)
                pred = att_model(x)
                #att_matrix = np.asarray([attributes[i] for i in train_labels])
                att_matrix = np.asarray([attributes for j in train_labels]).reshape(len(train_labels), nr_attributes)
                att_matrix = Variable(torch.from_numpy(att_matrix).float())
                combined = torch.cat((pred, att_matrix), 1)
                
                pred = rel_model(combined)
                # get loss
                print(loss_fn(pred, y))
                #print(pred[0], y[0])
                final_pred = np.argmax(pred.data.numpy(), axis=1)
                final_pred = [f+1 for f in final_pred]
                print(len(train_labels), len(final_pred))
                print("acc on train set", accuracy_score(train_labels, final_pred))
                
        
        x, y = Variable(torch.from_numpy(test.astype(float)).float()), Variable(torch.from_numpy(OH_test_labels).float(), requires_grad=False)
        pred = att_model(x)
        #att_matrix = np.asarray([attributes[i] for i in test_labels])
        att_matrix = np.asarray([attributes for j in test_labels]).reshape(len(test_labels), nr_attributes)
        att_matrix = Variable(torch.from_numpy(att_matrix).float())
        combined = torch.cat((pred, att_matrix), 1)
        
        pred = rel_model(combined)
        print(loss_fn(pred, y))
        final_pred = np.argmax(pred.data.numpy(), axis=1)
        final_pred = [f+1 for f in final_pred]
                
        acc = accuracy_score(test_labels, final_pred)
        print("acc on test set", acc)
        #print(final_pred, test_labels)
        print(final_pred, test_labels)
        confusion_matrix(final_pred, test_labels)
        result_row.append(acc)
    
    r = result_row
    result_row.append(np.average(r))
    result_row.append(np.std(r))
    
    results.append(result_row)

for r in results:
    print(r)
    





