# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 10:03:13 2018

@author: Christopher
"""

from sklearn.datasets import load_digits
import numpy as np

def flatten(_list):
    if len(_list) == 1:
        return _list[0]
    else:
        return _list[0] + flatten(_list[1:])
    

def one_hot(labels, nr_classes):
    _labels = []
    for l in labels:
        _labels.append([0] * nr_classes)
        _labels[-1][l] = 1
    return np.asarray(_labels)


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
        all_instances = data[str(c)]
        left, right = all_instances[0:int(len(all_instances) * 0.8)], all_instances[int(len(all_instances) * 0.8):]
        train.append(left)
        label_train.append([int(c)-1] * len(left)) # since the labels start at on, subtract 1
        test.append(right)
        label_test.append([int(c)-1] * len(right))
        
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
    
    
    with_class = [x for x in range(1,17) if x not in without]
    #(train, label_train), (test, label_test) = split_train_test(data, classes = list(range(1,16)))
    return split_train_test(data, classes = with_class)#(train, label_train), (test, label_test)
    

def load_hyp_spectral():
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
    return data









































