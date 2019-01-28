# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:32:12 2018

@author: Christopher
"""

import numpy as np
from load_data import load_hyp_spectral
from utils2 import save_object, heatmap
from utils import load_data
import torch

def extract_simple_features():
    """
    Before we use more complicated methods, like pretrained NN, lets start with simple features:
    - simply the average over 50 channels (400-405, 406-410 ect)
    """
    # load the data
#    from scipy.io import loadmat
#    
#    f = loadmat("data/indian_pines.mat")
#    raw_data = f['indian_pines']
#    
#    f = loadmat("data/indian_pines_gt.mat")
#    GT = f['indian_pines_gt']
#    
#    data = {}
#    
#    for row in range(len(GT)):
#        for col in range(len(GT[row,:])):
#            if str(GT[row,col]) in data:
#                data[str(GT[row,col])].append(raw_data[row, col, :])
#            else:
#                data[str(GT[row, col])] = []
#                data[str(GT[row, col])].append(raw_data[row, col, :])
    data = load_hyp_spectral()
    features = []
    for key in data.keys():
        averages = sum(data[key])
        averages = np.dot(averages, 1/len(data[key]))
        channels = np.split(averages, 20)
        row = [sum(c)/len(c) for c in channels]
        features.append([key] + row)
        
    print(features)
    with open("hyp_simple_features.txt", 'w') as f:
        for row in features:
            f.write(row[0] + ":")
            for nr in row[1:-1]:
                f.write(str(nr) + ",")
            f.write(str(row[-1]) + "\n")



def extract_abstract_features():
    """
    We can use pretrained networks to kinda extract more abstract features. The 
    problem is that they have a very specific input format: (3 x H x W), where 
    H and W are expected to be at least 224. The 3 obviously stands for the 3 rgb 
    channels.
    Thus we neet to restructure our hyp-spec data into that format.
    Idea 1: just use all bandwidths of one class and put them under each other
    Idea 2: put the average of a class N times under each other
    Idea 3: multiply the average bandwith of a class with itself, sothat a matrix 
    results.
    Idea 4: work in progress
    """
    # load data:
    data = load_hyp_spectral()
    # load the pre trained model, in this case we take VVG16, 
    # because it is trained on a wide range of different objects. We remove 
    # the last layer to get the high level features
    from keras.preprocessing import image
    from keras.applications.vgg16 import VGG16
    from keras.applications.vgg16 import preprocess_input
    model = VGG16(weights='imagenet', include_top=False)
    
#    img_path = 'IMG_8352.jpg'
#    img = image.load_img(img_path, target_size=(224, 224))
#    img_data = image.img_to_array(img)
#    print(img_data.shape)
#    img_data = np.expand_dims(img_data, axis=0)
#    print(img_data.shape)
#    img_data = preprocess_input(img_data)
#    print(img_data.shape)
#    vgg16_feature = model.predict(img_data)
    
    # Idea 1:
    if False:
        print("Idea 1")
        with open("abstract_features_idea1.txt", 'w') as f:
            for c in range(1,17):
                img = data[str(c)]
                # fill every bandwidth with 4 zeros to get a width of at leatst 224:
                img = [list(row) + [0]*4 for row in img]
                # we restrict ourselves on 500 samples, because using all crashed my pc really hard
                # also we will extend to 500 samples if there are not enough, otherwise the feature vector is empty
                if len(img) < 500:
                    img = [[row, row, row] for row in img] * int(500/len(img))
                else:
                    img = [[row, row, row] for row in img[0:500]]
                
                img_data = np.asarray(img)
                img_data = np.swapaxes(img_data, 1, 2)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)
                vgg16_feature = model.predict(img_data)
                vgg16_feature = np.reshape(vgg16_feature,(1,-1))
                f.write(str(c) + ":" + str(vgg16_feature[0][0]))
                for value in vgg16_feature[0][1:]:
                    f.write("," + str(value))
                f.write("\n")
    # Idea 2: this might be better, since Idea 1 maybe gave convolutional 
    # information between the sample, which is not there
    if False:
        print("Idea 2")
        with open("abstract_features_idea2.txt", 'w') as f:
            for c in range(1,17):
                img = data[str(c)]
                img = np.average(img, 0)
                # fill every bandwidth with 4 zeros to get a width of at leatst 224:
                img = np.append(img, [0,0,0,0])
                # we restrict ourselves on 500 samples, because using all crashed my pc really hard
                img = np.asarray([[img,img,img]] * 500)
                # rest: same as above
                img_data = np.asarray(img)
                img_data = np.swapaxes(img_data, 1, 2)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)
                vgg16_feature = model.predict(img_data)
                vgg16_feature = np.reshape(vgg16_feature,(1,-1))
                f.write(str(c) + ":" + str(vgg16_feature[0][0]))
                for value in vgg16_feature[0][1:]:
                    f.write("," + str(value))
                f.write("\n")
    # Idea 3: this is just a random idea, maybe some kind of de-composition is involved
    if False:
        print("Idea 3")
        with open("abstract_features_idea3.txt", 'w') as f:
            for c in range(1,17):
                img = data[str(c)]
                img = np.average(img, 0)
                # fill every bandwidth with 4 zeros to get a width of at leatst 224:
                img = np.append(img, [0,0,0,0])
                # now do weird vector multiplication:
                img = np.expand_dims(img, axis=0)
                img = np.matmul(np.transpose(img), img)
                # now add rgb channel:
                
                copy = np.zeros(shape=(224,224,3))
                copy[:,:,0] = img
                copy[:,:,1] = img
                copy[:,:,2] = img
                img_data = copy
                
                # rest: same as above
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)
                vgg16_feature = model.predict(img_data)
                vgg16_feature = np.reshape(vgg16_feature,(1,-1))
                f.write(str(c) + ":" + str(vgg16_feature[0][0]))
                for value in vgg16_feature[0][1:]:
                    f.write("," + str(value))
                f.write("\n")
    

def select_attributes(file):
    # load attribute data:
    # lets start with random attributes CxA, C is number of classes, A is number of attributes:
    # attributes = np.random.rand(16, 10)
    # or assume no attribues:
    attributes = np.zeros(shape=(16, 10))
    #import matplotlib.pyplot as plt
    #plt.imshow(digits['images'][104])
    #print(digits['target'][104])
    
    # load the attributes from the file
    attributes = {}
    with open(file, 'r') as f:
    #with open("hyp_simple_features.txt", 'r') as f:
        
        data = f.readlines()
        for line in data:
            label, features = line.split(":")
            values = [float(v) for v in features.replace("\n", "").split(",")]
            #print(values)
            attributes[int(label)] = values
    #print(len(attributes[0])) # <--53760 for abstract features, 20 for simple features
    sel_attributes = {}
    # Now remove all features that are 0 everywhere, bc they do nothing
    zero_features = list(range(53760))
    for key in attributes:
        print(key)
        for i, val in enumerate(attributes[key]):
            if val != 0:
                #print("keep", i)
                if i in zero_features:
                    zero_features.remove(i)
    print(len(zero_features))
    sel_attributes = {}
    for key in attributes:
        print(key)
        # turns out, the feature vectors are not the same length, so we take the minimum of 43008 or just the first 1000.
        sel_attributes[key] = [attributes[key][i] for i in range(1000) if i not in zero_features]
    # save the feature matrix, bc it runs forever:
    save_obj(sel_attributes, file.split(".")[0] + "_selected")
    # with this selection, we remove 43448 irrelevant features.


def perfect_features():
    features = np.eye(17)
    f = {}
    for index, row in enumerate(features):
        f[str(index)] = row
    save_object(f, "obj/perfect_features.pkl")


def pca_features():
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 10)
    train, train_labels, test, test_labels, _ = load_data(root_dir='./',dataset_name='PINE', balance=0, excluded_class=[],  scale_dataset=True, perc_split = [100,0,0])
    
    pca.fit(train)
    avgs = []
    for c in set(train_labels):
        row = []
        for pixel, label in zip(train, train_labels):
            if label == c:
                row.append(pixel)
        avgs.append(np.average(row, 0))
    print(pca.n_components_)
    print(pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_))
    #print(avgs)
    features = pca.transform(avgs)
    #print(features)
    #heatmap(features, x_size=1, y_size=0.4)
    f = {}
    for index, row in enumerate(features):
        f[str(index)] = row
    #save_object(f, "obj/pca_features.pkl")
    """
    the first 10 PCs explain 92.5% of the variance. We take them as features to keep the nr of features constant.
    """
    
    

#perfect_features()
#select_attributes('abstract_features_idea3.txt')

#extract_abstract_features()    
pca_features()    
