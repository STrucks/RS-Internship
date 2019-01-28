# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 10:03:13 2018

@author: Christopher
"""

from sklearn.datasets import load_digits
import numpy as np
import pickle
from utils import fromMAT, load_data
    
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
    
def flatten(_list):
    if len(_list) == 1:
        return _list[0]
    else:
        return _list[0] + flatten(_list[1:])


def one_hot2(labels, nr_classes):
    _labels = []
    for l in labels:
        #print(l)
        #print(int(l))
        _labels.append(np.zeros(shape=(nr_classes)))
        _labels[-int(1)][int(l)] = 1
    return np.asarray(_labels)


def one_hot(labels, nr_classes):
    _labels = []
    for l in labels:
        #print(l)
        #print(int(l))
        _labels.append(np.zeros(shape=(nr_classes)))
        _labels[-int(1)][int(l)-1] = 1
    return np.asarray(_labels)


def load_MNIST_raw():
    digits = load_digits()
    data = [np.reshape(img,newshape=(1,8*8)) for img in digits['images']]
    data = [d[0] for d in data]
    labels = digits['target']
    return data, labels
    

def load_MNIST():
    digits = load_digits()
    data = [np.reshape(img,newshape=(1,8*8)) for img in digits['images']]
    data = [d[0] for d in data]
    labels = digits['target']
    
    train, test = np.asarray(data[0:int(0.8*len(data))]), np.asarray(data[int(0.8*len(data)):])
    train_labels , test_labels = np.asarray(labels[0:int(0.8*len(labels))]), np.asarray(labels[int(0.8*len(labels)):])
    OH_train_labels, OH_test_labels = one_hot(np.asarray(labels[0:int(0.8*len(labels))]), 10), one_hot(np.asarray(labels[int(0.8*len(labels)):]), 10)
    
    return (train, train_labels, OH_train_labels), (test, test_labels, OH_test_labels)


def load_MNIST_ZSL(without=[9]):
    digits = load_digits()
    data = [np.reshape(img,newshape=(1,8*8)) for img in digits['images']]
    data = [d[0] for d in data]
    labels = digits['target']
    
    train, test = data[0:int(0.8*len(data))], data[int(0.8*len(data)):]
    train_labels , test_labels = labels[0:int(0.8*len(labels))], list(labels[int(0.8*len(labels)):])
    OH_train_labels, OH_test_labels = list(one_hot(np.asarray(labels[0:int(0.8*len(labels))]), 10)), list(one_hot(np.asarray(labels[int(0.8*len(labels)):]), 10))
    if len(without) > 0:
        remove = np.where(train_labels == without[0])[0]
        remove = list(reversed(remove))
        for i in remove:
            test.append(train[i])
            test_labels.append(train_labels[i])
            OH_test_labels.append(OH_train_labels[i])
    return (np.asarray(train), np.asarray(train_labels), np.asarray(OH_train_labels)), (np.asarray(test), np.asarray(test_labels), np.asarray(OH_test_labels))


def split_train_test(data, classes=[1,2,3,4,5,6,7,11]):
    # this loses context information, like neighbours ect
    """
    This function divides the data into a testing and a training set. For all classes specified with 'classes' 80% is put
    into the training and 20% in the testing set by default. All remaining instances of the other classes are put into the
    testing set.
    :param data: a data dictionary with data['1'] all entries of class 1
    :param classes: classes in training set
    :return: (train, label_train), (test, label_test)
    """

    train = []
    label_train = []
    test = []
    label_test = []
    for c in classes:
        if c == 0: continue
        all_instances = data[str(c)]
        left, right = all_instances[0:int(len(all_instances) * 0.8)], all_instances[int(len(all_instances) * 0.8):]
        train.append(left)
        label_train.append([int(c)] * len(left)) # since the labels start at on, subtract 1
        test.append(right)
        label_test.append([int(c)] * len(right))
        
    others = [str(x) for x in range(1,17) if x not in classes]
    for o in others:
        test.append(data[str(o)])
        label_test.append([int(o)-1] * len(data[str(o)]))
    
    
    train = np.asarray(flatten(train))
    test = np.asarray(flatten(test))
    label_train = np.asarray(flatten(label_train))
    label_test = np.asarray(flatten(label_test))
    
    
    OH_train_labels = one_hot(label_train, 16)
    OH_test_labels = one_hot(label_test, 16)
    
    return (train, label_train, OH_train_labels), (test, label_test, OH_test_labels)


def load_hyp_spectral_splitted(without = [15, 16]):
    from scipy.io import loadmat
    
    f = loadmat("data/indian_pines_mh.mat")
    raw_data = f['indian_pines']
    
    f = loadmat("data/indian_pines_gt.mat")
    GT = f['indian_pines_gt']
    
    #raw_data, GT = balance(raw_data, GT)
    
    data = {}
    
    for row in range(len(GT)):
        for col in range(len(GT[row,:])):
            if GT[row,col] == 0: continue
            if str(GT[row,col]) in data:
                data[str(GT[row,col])].append(raw_data[row, col, :])
            else:
                data[str(GT[row, col])] = []
                data[str(GT[row, col])].append(raw_data[row, col, :])
    
    
    with_class = [x for x in range(1,17) if x not in without]
    #(train, label_train), (test, label_test) = split_train_test(data, classes = list(range(1,16)))
    return split_train_test(data, classes = with_class)#(train, label_train), (test, label_test)
    

def balance(data, GT, max_nr = 100):
    # select random cases of each class
    print(GT)
    data = []
    for i in range(len(GT)):
        for j in range(len(GT[i])):
            continue
    
#load_hyp_spectral_splitted(without=[])

def load_hyp_spectral():
    from scipy.io import loadmat
    f = loadmat("data/indian_pines.mat")
    raw_data = f['indian_pines']
    
    f = loadmat("data/indian_pines_gt.mat")
    GT = f['indian_pines_gt']
    
    data = {}
    
    for row in range(len(GT)):
        for col in range(len(GT[row,:])):
            if GT[row,col] == 0:
                continue
            if str(GT[row,col]) in data:
                data[str(GT[row,col])].append(raw_data[row, col, :])
            else:
                data[str(GT[row, col])] = []
                data[str(GT[row, col])].append(raw_data[row, col, :])
    return data

def load_hyp_spectral_preprocessed(perc_split = [100,0,0]):
    X, Y, X_test, y_test, mapping = load_data(root_dir='./',dataset_name='PINE', balance=100, scale_dataset=True, perc_split = perc_split)
    data = {}
    
    for x, y in zip(X,Y):
        y = int(y)+1
        if y in data:
            data[y].append(x)
        else:
            data[y] = list([x])
    #print(data)
               
    
#    for row in range(len(GT)):
#        for col in range(len(GT[row,:])):
#            if GT[row,col] == 0:
#                continue
#            if str(GT[row,col]) in data:
#                data[str(GT[row,col])].append(raw_data[row, col, :])
#            else:
#                data[str(GT[row, col])] = []
#                data[str(GT[row, col])].append(raw_data[row, col, :])
    return data


def load_attributes(file, length = 0):
    data = load_obj(file)
    if length > 0:
        for key in data:
            data[key] = data[key][0:length]
    return data


def preprocess(exclude = [0]):
    train_set, _, test_set, mapping, scaler  = fromMAT(path="data/indian_pines", varname="indian_pines", perc_split=[80,0,20],
                                                       scale_dataset = True, excluded_class = exclude, augmentation_type = 'l')
    """
    Questions: 
    why is the size not changing after label augmentation?
    excluded class removes the classes from BOTH training and testing phase?
    """
    
    OH_train = one_hot(train_set[1], 16)
    OH_test = one_hot(test_set[1], 16)
    
    return (train_set[0], train_set[1], OH_train), (test_set[0], test_set[1], OH_test)


def load_hyp_spectral_splitted_preprocessed(without = [15, 16]):
    return preprocess(without)



#load_hyp_spectral_preprocessed()



























