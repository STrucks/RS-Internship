from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np


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


def split_train_test(classes=[1,2,3,4,5,6,7]):
    # this loses context information, like neighbours ect
    """
    This function divides the data into a testing and a training set. For all classes specified with 'classes' 80% is put
    into the training and 20% in the testing set by default. All remaining instances of the other classes are put into the
    testing set.
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
        label_train.append(c)
        test.append(right)
        label_test.append(c)

    others = [str(x) for x in range(1,17) if x not in classes]
    for o in others:
        test.append(data[str(o)])
        label_test.append(o)
    return (train, label_train), (test, label_test)


