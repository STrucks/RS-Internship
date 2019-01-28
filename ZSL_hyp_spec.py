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
from matplotlib import pyplot as plt
from load_data import load_hyp_spectral_splitted, load_attributes, load_hyp_spectral_splitted_preprocessed, preprocess, one_hot, one_hot2
from utils2 import confusion_matrix
from utils import load_data
#from utils import load_data



#(train, train_labels, OH_train_labels), (test, test_labels, OH_test_labels) = load_MNIST()
#(train, train_labels, OH_train_labels), (test, test_labels, OH_test_labels) = preprocess(exclude = [0])

#(train, train_labels, OH_train_labels), (test, test_labels, OH_test_labels) = load_hyp_spectral_splitted()
general_excluded_cl = [1., 7., 9., 13., 16.]
excluded_cl = [2.,3.,4.,5.,6.]

# for some reason, the load data function swaps the train and test set: train set is 2652 instances and test is 8923???? I will thus swap these
test, test_labels, train, train_labels, _ = load_data(root_dir='./',dataset_name='PINE', balance=100, excluded_class=[0],  scale_dataset=True)
OH_train_labels = one_hot2(train_labels,16)
#_, _, test, test_labels, _ = load_data(root_dir='./',dataset_name='PINE', balance=100, excluded_class=[0],  scale_dataset=True, perc_split = [])
OH_test_labels = one_hot2(test_labels,16)

# problem with the load data function: excluded class removes the classes, but changes the label, so 1-> 0, 9->8 ect.
# remove some class instances: also, balancing seems to not be working
copy_train = []
copy_train_labels = []
copy_OH_train = []
copy_test = []
copy_test_labels = []
copy_OH_test = []
print(train_labels.shape, test_labels.shape)
freq = {}
for index, value in enumerate(train_labels):
    # the value + 1 is to correct for the removal of class 0
    if (value+1) not in general_excluded_cl and (value+1) not in excluded_cl:
        if value not in freq:
            freq[value] = 1
        else:
            if freq[value] >= 200:
                continue
            freq[value] += 1
        copy_train.append(train[index])
        copy_train_labels.append(value)
        copy_OH_train.append(OH_train_labels[index])

freq = {}
for index, value in enumerate(test_labels):
    # the value + 1 is to correct for the removal of class 0
    if (value+1) in general_excluded_cl:
        continue
    elif (value+1) in freq:
        if freq[value+1] >= 40:
                continue
        else:
            freq[value+1] += 1
    else:
        freq[value+1] = 1
           
    copy_test.append(test[index])
    copy_test_labels.append(value)
    copy_OH_test.append(OH_test_labels[index])

exc_test = []
exc_test_labels = []
freq = {}
for index, value in enumerate(test_labels):
    # the value + 1 is to correct for the removal of class 0
    if (value+1) in excluded_cl:
        if value not in freq:
            freq[value] = 1
        else:
            if freq[value] >= 40:
                continue
            freq[value] += 1
        exc_test.append(test[index])
        exc_test_labels.append(value)


exc_test_OH = one_hot2(exc_test_labels,16)




exc_test, exc_test_labels = np.asarray(exc_test), np.asarray(exc_test_labels)

test, test_labels, train, train_labels = np.asarray(copy_test), np.asarray(copy_test_labels), np.asarray(copy_train), np.asarray(copy_train_labels)
OH_train_labels, OH_test_labels = np.asarray(copy_OH_train), np.asarray(copy_OH_test)


#import sys
#sys.exit()


#print(train.shape, train_labels.shape, test.shape, test_labels.shape)
print(OH_train_labels)
print(min(train_labels), max(train_labels),min(test_labels))
# number of neurons in each layer
input_num_units = 220
hidden_num_units = 100
output_num_units = 16
# set remaining variables 
epochs = 10
batch_size = 128
learning_rate = 0.0001
nr_batches = len(train)#8194#241
nr_attributes = 10 * 16
#'hyp_simple_features.txt', 
att_data = ['abstract_features_idea1.txt','abstract_features_idea2.txt', 'abstract_features_idea3.txt']
att_data = ['autoencoder_features_10_e50']
#att_data = ['pca_features']
# att_data = ['abstract_features_idea1']


trials = 1
results = []


#(train, train_labels, OH_train_labels), (test, test_labels, OH_test_labels) = load_MNIST()
#(train, train_labels, OH_train_labels), (test, test_labels, OH_test_labels) = preprocess(exclude = [0])

#(train, train_labels, OH_train_labels), (test, test_labels, OH_test_labels) = load_hyp_spectral_splitted()
general_excluded_cl = [1., 7., 9., 13., 16.]
excluded_cl = [2.,3.,4.,5.,6.]

#for ex_cl in range(len(excluded_cl)):

# for some reason, the load data function swaps the train and test set: train set is 2652 instances and test is 8923???? I will thus swap these
test, test_labels, train, train_labels, _ = load_data(root_dir='./',dataset_name='PINE', balance=100, excluded_class=[0],  scale_dataset=True)
OH_train_labels = one_hot2(train_labels,16)
#_, _, test, test_labels, _ = load_data(root_dir='./',dataset_name='PINE', balance=100, excluded_class=[0],  scale_dataset=True, perc_split = [])
OH_test_labels = one_hot2(test_labels,16)

# problem with the load data function: excluded class removes the classes, but changes the label, so 1-> 0, 9->8 ect.
# remove some class instances: also, balancing seems to not be working
copy_train = []
copy_train_labels = []
copy_OH_train = []
copy_test = []
copy_test_labels = []
copy_OH_test = []
print(train_labels.shape, test_labels.shape)
freq = {}
for index, value in enumerate(train_labels):
    # the value + 1 is to correct for the removal of class 0
    if (value+1) not in general_excluded_cl and (value+1) not in excluded_cl:
        if value not in freq:
            freq[value] = 1
        else:
            if freq[value] >= 200:
                continue
            freq[value] += 1
        copy_train.append(train[index])
        copy_train_labels.append(value)
        copy_OH_train.append(OH_train_labels[index])

freq = {}
for index, value in enumerate(test_labels):
    # the value + 1 is to correct for the removal of class 0
    if (value+1) in general_excluded_cl:
        continue
    elif (value+1) in freq:
        if freq[value+1] >= 40:
                continue
        else:
            freq[value+1] += 1
    else:
        freq[value+1] = 1
           
    copy_test.append(test[index])
    copy_test_labels.append(value)
    copy_OH_test.append(OH_test_labels[index])

exc_test = []
exc_test_labels = []
freq = {}
for index, value in enumerate(test_labels):
    # the value + 1 is to correct for the removal of class 0
    if (value+1) in excluded_cl:
        if value not in freq:
            freq[value] = 1
        else:
            if freq[value] >= 40:
                continue
            freq[value] += 1
        exc_test.append(test[index])
        exc_test_labels.append(value)


exc_test_OH = one_hot2(exc_test_labels,16)




exc_test, exc_test_labels = np.asarray(exc_test), np.asarray(exc_test_labels)

test, test_labels, train, train_labels = np.asarray(copy_test), np.asarray(copy_test_labels), np.asarray(copy_train), np.asarray(copy_train_labels)
OH_train_labels, OH_test_labels = np.asarray(copy_OH_train), np.asarray(copy_OH_test)


#import sys
#sys.exit()


#print(train.shape, train_labels.shape, test.shape, test_labels.shape)
print(OH_train_labels)
print(min(train_labels), max(train_labels),min(test_labels))



for file in att_data[0:1]:
    result_row = []
    print(file)
    #load attributes:
    attributes = load_attributes(file, 10)
    attributes = np.asarray([np.asarray(attributes[str(key)]) for key in range(1,17)])
    attributes = np.reshape(attributes, newshape=(1,-1))
    print(attributes.shape)
    acc_miss = []
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
        rel_optimizer = torch.optim.Adam(rel_model.parameters(), lr=learning_rate)
        att_optimizer = torch.optim.Adam(att_model.parameters(), lr=learning_rate)
       
        #print(len(train))
        losses = []
        print("train")
        for epoch in range(epochs):# how many times we iterate over the whole dataset
            train_batch = np.split(train,nr_batches)
            label_batch = np.split(train_labels,nr_batches)
            OH_batch = np.split(OH_train_labels,nr_batches)
            epoch_loss = 0
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
                epoch_loss += loss.data
                # perform backpropagation
                att_model.zero_grad()
                rel_model.zero_grad()
                loss.backward()
                att_optimizer.step()
                rel_optimizer.step()
                
                
            #print(epoch, "loss", epoch_loss)
            losses.append(epoch_loss/nr_batches)
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
                #print(loss_fn(pred, y))
                #print(pred[0], y[0])
                final_pred = np.argmax(pred.data.numpy(), axis=1)
                #final_pred = [f+1 for f in final_pred]
                #print(len(train_labels), len(final_pred))
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
        #final_pred = [f+1 for f in final_pred]
                
        acc = accuracy_score(test_labels, final_pred)
        print("acc on test set", acc)
        #print(final_pred, test_labels)
        #print(final_pred, test_labels)
        #confusion_matrix(final_pred, test_labels)
        result_row.append(acc)
        """
        filtered_test = []
        filtered_test_label = []
        filtered_test_label_OH = []
        
        for index in range(len(test)):
            if test_labels[index] in excluded_cl:
                filtered_test.append(test[index])
                filtered_test_label.append(test_labels[index])
                filtered_test_label_OH.append(OH_test_labels[index])
        filtered_test = np.asarray(filtered_test)
        filtered_test_label = np.asarray(filtered_test_label)
        filtered_test_label_OH = np.asarray(filtered_test_label_OH)
        print("filtered", filtered_test.shape, filtered_test_label.shape)
        """
        if len(excluded_cl) > 0:
            x, y = Variable(torch.from_numpy(exc_test.astype(float)).float()), Variable(torch.from_numpy(exc_test_OH).float(), requires_grad=False)
            pred = att_model(x)
            #att_matrix = np.asarray([attributes[i] for i in test_labels])
            att_matrix = np.asarray([attributes for j in exc_test_labels]).reshape(len(exc_test_labels), nr_attributes)
            att_matrix = Variable(torch.from_numpy(att_matrix).float())
            combined = torch.cat((pred, att_matrix), 1)
            
            pred = rel_model(combined)
            print(loss_fn(pred, y))
            final_pred = np.argmax(pred.data.numpy(), axis=1)
            #final_pred = [f+1 for f in final_pred]
            
            acc2 = accuracy_score(exc_test_labels, final_pred)
            plt.subplot(2,2,1)
            confusion_matrix(final_pred, exc_test_labels, image=True)
            print("acc on excluded set", acc2)
        
        
        
        plt.plot(losses)
        plt.show()
        
        
    
    r = result_row
    result_row.append(np.average(r))
    result_row.append(np.std(r))
    
    results.append(result_row)

for r in results:
    print(r)
    



"""
with all classes:
[0.4079822616407982, 0.44789356984478934, 0.3325942350332594, 0.3961566888396157, 0.0414027008844373]
trials = 3, epochs = 100
with no class 2:
[0.2638580931263858, 0.2727272727272727, 0.29933481152993346, 0.278640059127864, 0.013055078885924382]
with no class 2,3:
[0.31263858093126384, 0.3902439024390244, 0.3968957871396896, 0.3665927568366593, 0.03312364499674289]
with no class 2,3,4:
[0.31263858093126384, 0.3902439024390244, 0.3968957871396896, 0.3665927568366593, 0.03312364499674289]

"""





