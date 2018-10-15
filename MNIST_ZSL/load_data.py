# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 10:03:13 2018

@author: Christopher
"""

from sklearn.datasets import load_digits
import numpy as np

def one_hot(labels, nr_classes):
    _labels = []
    for l in labels:
        _labels.append([0]*nr_classes)
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
