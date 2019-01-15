#!/usr/bin/python


from __future__ import print_function
import sys
import os
import shutil
import traceback
import signal
import time
import itertools
import glob
import re
#from PIL import Image
import theano
import theano.tensor as T
from theano import function
from math import sqrt,ceil,floor,exp
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.cross_validation import StratifiedShuffleSplit
from collections import Counter
from itertools import combinations
from random import shuffle, randint
#from lasagne.regularization import regularize_layer_params, l2, l1
#from nolearn.lasagne import visualize
#import lasagne
#from lasagne.objectives import aggregate
#from lasagne.layers import get_output, get_output_shape, get_all_layers, get_all_param_values,get_all_params
#from lasagne.init import Initializer
#from lasagne import layers
import numpy as np
from scipy.special import binom
from scipy.ndimage.filters import gaussian_filter1d
from scipy import signal
import scipy.stats as st
from scipy.io import savemat, loadmat

from sklearn import preprocessing


import pickle

import random

"""
Author: Jacopo Aquarelli (??)
"""


def float32(k):
    """Converts a number or an array of numbers into the np.float32 format
    Parameters
    ----------
    k : array_like or number
        The data to be converted.
    Returns
    -------
    np ndarray or number
        The converted array/number
    """
    return np.cast['float32'](k)


def l2_paired(x):
  """Spectral smoothing
    Applies a modified L2 norm to a 1D vector that takes 
    into account the locality of the information
    Parameters
    ----------
    x : theano tensor 
        The input tensor.
    Returns
    -------
    theano tensor
        The output tensor
  """
  shapes=x.shape.eval()
  mask=np.eye(shapes[-1])
  mask[-1,-1]=0
  rolled=T.roll(x,-1,axis=len(shapes)-1)
  return T.sum((x - T.dot(rolled,mask))**2)


def my_objective(layers, loss_function, target, lamda1, lamda2, aggregate = True, # changed aggregate = aggregate to aggregate = True
                 deterministic=False, get_output_kw=None):
    """Custom objective function for Nolearn that include 2 different
    type of regularization terms
    Parameters
    ----------
    layers : array of Lasagne layers
        All the layers of the neural network
    loss_function : function
        The loss function to use
    lamda1 : float
        Constant for the L2 regularizaion term
    lamda2 : float
        Constant for the paired L2 regularizaion term
    aggregate : function
        Lasagne function to aggregate an element
        or item-wise loss to a scalar loss.
    deterministic : boolean
    
    Returns
    -------
    float
        The aggregated loss value
    """
    if get_output_kw is None:
        get_output_kw = {}
    net_out = get_output(
        layers[-1],
        deterministic=deterministic,
        **get_output_kw
        )
    hidden_layers = layers[1:-1]
    losses = loss_function(net_out, target) #+ (lamda1/100)*regularize_layer_params(layers[-1], l2)
    if not deterministic:
      for i,h in enumerate(hidden_layers):
        zeros = np.zeros(i).astype(int).astype(str)                
        denom = '10' +  ''.join(zeros) 
        if isinstance(h, (int,str)):
          if 'input' in h:
            continue
          h_layer=hidden_layers[h]
        else:
          h_layer=h
        print('AAAAAAAAAAAAAAAAAA',lamda1,lamda2)
        losses = losses + (i/float(denom)) * (lamda1*regularize_layer_params(h_layer, l2) + lamda2*regularize_layer_params(h_layer, l2_paired)) 
    return aggregate(losses)
  

class EarlyStopping(object):
    """Class to handle early stopping in the Nolearn-Lasagne estimator"""
    def __init__(self, patience=100,verbose=False):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
        
        self.verbose=verbose

    def __call__(self, nn, train_history):
        #print(np.mean(nn.layers_['conv1d'].W.get_value()))
        current_valid = train_history[-1]['valid_loss']
        #print(train_history[-1])
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            #if self.verbose:
              #print("Early stopping.")
              #print("Best valid loss was {:.6f} at epoch {}.".format(self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()


def saveVar(var,name,path='.' + os.sep + 'saved_networks' + os.sep ):
  """Save a python variable into a .pickle file
  Parameters
  ----------
  var : python variable
    the python variable to save
  name : str
    the name of the .pickle output file
  path : str
    the directory where to save the .pickle file
  Returns
  -------
  nothing
  
  """
  try:
    if not path.endswith('' + os.sep + ''):
      path+='' + os.sep + ''
    f=open(path + name + '.pickle', 'w')
    pickle.dump(var, f)
    print('Writing variable:',name,'in path:',path)
  except IOError:
    print('Error writing variable:',name,'in path:',path)
    
    
def loadVar(filename,path='.' + os.sep + 'saved_networks' + os.sep,verbose=True):
  """Load a python variable from a .pickle file
  Parameters
  ----------
  name : str
    the name of the .pickle output file
  path : str
    the directory where the .pickle file is located
  Returns
  -------
  var : python variable
    the loaded python variable
  """
  try:
    if not filename.endswith('.pickle'):
      filename+='.pickle'
    if not path.endswith('' + os.sep + ''):
      path+='' + os.sep + ''
    f=open(path + filename , 'r')
    var=pickle.load(f)
    if verbose:
        print('Reading filename:',filename,'in path:',path)
    return var
  except IOError:
    print('Error reading filename:',filename,'in path:',path)
    return np.array([])
  

def remove_samples_by_class(tmp_data,tmp_gt,excluded_class):
    """Function that remove samples which class is included in the 'excluded_class' array"""
    data=tmp_data[np.logical_not(np.in1d(tmp_gt,excluded_class)),:]
    data_gt=tmp_gt[np.logical_not(np.in1d(tmp_gt,excluded_class))]
    new_labels=np.unique(data_gt)
    init_gt=data_gt
    for i,c in enumerate(new_labels):
        data_gt[init_gt==c]=i
    return data,data_gt,np.logical_not(np.in1d(tmp_gt,excluded_class))


def rebuild_from_excluded(y,mapping,orig_pixel_num):
    """Function that rebuilds the original picture from an input with removed samples"""
    ret_y=np.zeros((orig_pixel_num))
    j=0
    #print(ret_y.shape)
    for i in range(orig_pixel_num):
        if i in mapping:
            ret_y[mapping[j]]=y[j]
            j=j+1
        else:
            ret_y[i]=-1
    return ret_y


def fromMAT(path="dataset.mat",varname="data",perc_split=[],seed=-1,scale_dataset=False,excluded_class=[],balance=0,window=1,augmentation_type=''):
    """Load data and ground-truth from a .mat file
    Parameters
    ----------
    path : str
        the full path of the .mat input dataset file:
        the ground-truth file name is assumed to
        have a '_gt' suffix
    varname : str
        the name of the dataset variable inside the .mat file:
        the variable inside the ground-truth file is assumed to
        have a '_gt' suffix
    perc_split : array_like
        the percentage of samples to include for train,validatio,test
    seed : int
        seed for shuffling the data (-1 means no shuffle, 0 random seed)
    scale_dataset : boolean
        whether to normalize or not the dataset between 0 and 1
    augmentation_type : str
        which data augmentation to do (a, n l,al: a=noise, l=label based data augmentation)
    Returns
    -------
    (data_train,label_train) : pair of array_like elements
        dataset and labels for the train set
    (data_valid,label_valid) : pair of array_like elements
        dataset and labels for the validation set
    (data_test,label_test) : pair of array_like elements
        dataset and labels for the test set
    mapping : an array that keep track of the samples random shuffling
              depending on the input seed
    """
    if len(perc_split)==0:
        perc_split=[100,0,0]
    kernel_type=''
    data_patient=[]
    data_augment_factor=1+(1 if augmentation_type=='standard' or augmentation_type=='all' else 0)
    label_augmentation_type=(augmentation_type=='label' or augmentation_type=='separate' or augmentation_type=='all')
    separate_smoothed_traintest=(augmentation_type=='separate')
    separate_smoothed_traintest=False
    mapping_2d=[]
    if isinstance(balance,str):
        if balance.count('s')>0:
            separate_smoothed_traintest=True
            label_augmentation_type=True
        if balance.count('l')>0:
            label_augmentation_type=True
        data_augment_factor=balance.count('a')+1
        if balance.count('.')>0:
            balance=float(re.findall("\d+\.\d+", balance)[0])
        else:
            balance=int(filter(str.isdigit, balance))
    if isinstance(window,str):
        kernel_type=''.join(i for i in window if not i.isdigit())
        window=int(filter(str.isdigit, window))
    try:
        try:
            data=loadmat(path+'.mat')[varname+'_noise']
        except:
            data=loadmat(path+'.mat')[varname]
        data_orig=data
        data_gt=loadmat(path+'_gt.mat')[varname+'_gt']
        orig_shape=data_gt.shape
        mapping_2d=create2d1dmap(data_gt.shape)
        excluded_idx=np.zeros(data_gt.shape)
        excluded_idx=np.full(data_gt.shape, False, dtype=bool)
        for c in excluded_class:
            excluded_idx=excluded_idx+(data_gt==c)

        mapping_2d=mapping_2d[np.logical_not(excluded_idx.flatten())]

        if window>1:
            data=patch_preprocessing(data,sizes=window,kernel_type=kernel_type,excluded_idx=excluded_idx)


        if data_augment_factor>1:
            augmented_data=noise_data_augmentation(data,data_gt,alphas=[1,0],beta=0.01,factor=data_augment_factor)

            augmented_data_gt=np.zeros((data_augment_factor,data_gt.shape[0],data_gt.shape[1]))
            for i in range(data_augment_factor):

                augmented_data_gt[i,:,:]=data_gt[:,:]

        else:
            augmented_data=data.reshape((1,data.shape[0],data.shape[1],data.shape[2]))
            augmented_data_gt=data_gt.reshape((1,data.shape[0],data.shape[1]))
        augmented_data=np.append(augmented_data,data_orig.reshape((1,data_orig.shape[0],data_orig.shape[1],data_orig.shape[2])),axis=0)
        augmented_data_gt=np.append(augmented_data_gt,data_gt.reshape((1,data_gt.shape[0],data_gt.shape[1])),axis=0)
        data_augment_factor=data_augment_factor+1           
            
    except IOError:
        print("The file '",path,".mat' or '",path,"_gt.mat' doesn't exist")
        sys.exit(1)
    except NotImplementedError:
        import h5py
        data = np.array(h5py.File(path+'.mat').get(varname)).T
        data_gt = np.array(h5py.File(path+'_gt.mat').get(varname+'_gt')).T
        excluded_idx=np.zeros(data_gt.shape)
        excluded_idx=np.full(data_gt.shape, False, dtype=bool)
        for c in excluded_class:
            excluded_idx=excluded_idx+(data_gt==c)
        if window>1:
            data=patch_preprocessing(data,sizes=window,excluded_idx=excluded_idx)            

    #Hyperspectral data to 1D spectra
    if len(data.shape)==3:
        tmp_data=np.zeros((data.shape[0]*data.shape[1],data.shape[2]))
        tmp_gt=np.zeros(data.shape[0]*data.shape[1])
        k=0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                tmp_data[k,:]=data[i,j,:]
                tmp_gt[k]=data_gt[i,j]
                k+=1
        if data_augment_factor>1:
            tmp_data_augmented=np.zeros((augmented_data.shape[0],augmented_data.shape[1]*augmented_data.shape[2],augmented_data.shape[3]))
            tmp_gt_augmented=np.zeros((augmented_data.shape[0],augmented_data.shape[1]*augmented_data.shape[2]))
            for h in range(data_augment_factor):
                k=0
                for i in range(augmented_data.shape[1]):
                    for j in range(augmented_data.shape[2]):

                        tmp_data_augmented[h,k,:]=augmented_data[h,i,j,:]
                        tmp_gt_augmented[h,k]=augmented_data_gt[h,i,j]
                        k+=1
        if len(excluded_class)>0:
            
            data,data_gt,excluded_mapping=remove_samples_by_class(tmp_data,tmp_gt,excluded_class)
            if data_augment_factor>1:
                augmented_data=np.zeros((data_augment_factor,data.shape[0],data.shape[1]))
                augmented_data_gt=np.zeros((data_augment_factor,data_gt.shape[0]))
                for h in range(data_augment_factor):
                    a,b,_=remove_samples_by_class(tmp_data_augmented[h],tmp_gt_augmented[h],excluded_class)

                    augmented_data[h] = a
                    augmented_data_gt[h,:] = b[:]
   

        else:
            data=tmp_data
            data_gt=tmp_gt

            if data_augment_factor>1:
                augmented_data=tmp_data_augmented
                augmented_data_gt=tmp_gt_augmented
        try:
            
            data_patient=loadmat(path+'_patient.mat')[varname+'_patient'].flatten()

            data_patient=data_patient.flatten()[np.logical_not(excluded_idx).flatten()]
            
        except:
            pass
    
    if isinstance(seed, list) or type(seed).__module__ == np.__name__:
      data=data[seed,:]
      data_gt=data_gt[seed]
    elif int(seed) >=0:
      seed=int(seed)
      if seed>0:
        np.random.seed(seed)
      mapping=np.random.permutation(np.arange(data.shape[0]))
      mapping_2d=mapping_2d[mapping]
      data=data[mapping]
      data_gt=data_gt[mapping]
      if data_augment_factor>1:
        for h in range(data_augment_factor):
            tmp_aug_data=augmented_data[h]
            augmented_data[h,:,:]=tmp_aug_data[mapping]
            tmp_aug_data=augmented_data_gt[h]
            augmented_data_gt[h,:]=tmp_aug_data[mapping]

    else:
        mapping=np.arange(data.shape[0])

    train_min=0
    train_max=train_min + int(ceil(float(data.shape[0]*perc_split[0])/100.0))
    if train_max>=data.shape[0]:
      train_max=train_max-((train_max+1)%data.shape[0])          
    validation_min=train_max
    validation_max=validation_min+int(ceil(float(data.shape[0]*perc_split[1])/100.0))
    if validation_max>=data.shape[0]:
      validation_max=validation_max-((validation_max+1)%data.shape[0])
    test_min=validation_max
    test_max=test_min+int(ceil(float(data.shape[0]*perc_split[2])/100.0))
    if test_max>=data.shape[0]:
      test_max=test_max-(test_max%data.shape[0])
    if np.amax(balance)>0:
        
        if label_augmentation_type:

            idx_keep,mod_data_gt,excl_neigh=label_augmentation(data_gt,balance,orig_shape,mapping,excluded_mapping,mapping_2d,window,separate_smoothed_traintest)
            data_train=data[idx_keep]
            label_train=mod_data_gt[idx_keep]

            if data_augment_factor>1:
                for h in range(1,data_augment_factor):
                    tmp_data_aug=augmented_data[h]
                    data_train=np.append(data_train,tmp_data_aug[idx_keep],axis=0)
                    tmp_data_aug=augmented_data_gt[h]
                    label_train=np.append(label_train,tmp_data_aug[idx_keep])
            if separate_smoothed_traintest:
                data_valid=data[np.delete(np.arange(data.shape[0]),np.append(idx_keep,excl_neigh),axis=0)]
                label_valid=mod_data_gt[np.delete(np.arange(data.shape[0]),np.append(idx_keep,excl_neigh),axis=0)]
            else:

                
                data_valid=data[np.delete(np.arange(data.shape[0]),excl_neigh,axis=0)]
                label_valid=mod_data_gt[np.delete(np.arange(data.shape[0]),excl_neigh,axis=0)]

        else:
            idx_keep=balanceSubset(data_gt,balance,mapping_2d,window,True,separate_smoothed_traintest)
            data_train=data[idx_keep]
            label_train=data_gt[idx_keep]
            if data_augment_factor>1:
                for h in range(1,data_augment_factor):
                    tmp_data_aug=augmented_data[h]
                    data_train=np.append(data_train,tmp_data_aug[idx_keep],axis=0)
                    tmp_data_aug=augmented_data_gt[h]
                    label_train=np.append(label_train,tmp_data_aug[idx_keep])
            data_valid=data[np.delete(np.arange(data.shape[0]),idx_keep,axis=0)]
            label_valid=data_gt[np.delete(np.arange(data.shape[0]),idx_keep,axis=0)]
            mapping=np.append(mapping[idx_keep],np.append(mapping[idx_keep],mapping[np.delete(np.arange(data.shape[0]),idx_keep,axis=0)]))
        data_test=np.array([])
        label_test=np.array([])

    elif perc_split[1]==0:
        data_train=data[train_min:train_max+1]
        label_train=data_gt[train_min:train_max+1]
        data_valid=data[validation_min:validation_max]
        label_valid=data_gt[validation_min:validation_max]
        
        data_test=data[test_min:test_max]
        label_test=data_gt[test_min:test_max] # there was a bug: label_test=data[test_min:test_max] changed to label_test=data_gt[test_min:test_max]
    elif perc_split[2]==0:
        data_train=data[train_min:train_max]
        label_train=data_gt[train_min:train_max]
        data_valid=data[validation_min:validation_max+1]
        label_valid=data_gt[validation_min:validation_max+1]
        data_test=data[test_min:test_max]
        label_test=data_gt[test_min:test_max]

    else:
        data_train=data[train_min:train_max]
        label_train=data_gt[train_min:train_max]
        data_valid=data[validation_min:validation_max]
        label_valid=data_gt[validation_min:validation_max]
        data_test=data[test_min:test_max]
        label_test=data_gt[test_min:test_max]

    if scale_dataset:
      scaler = preprocessing.MinMaxScaler() 
      if data_train.shape[0]>0:
        data_train=scaler.fit_transform(data_train)
      if data_valid.shape[0]>0:
        data_valid=scaler.fit_transform(data_valid) # change
        #data_valid=scaler.transform(data_valid)
      if data_test.shape[0]>0:
        data_test=scaler.transform(data_test)
    else:
        scaler=None
    return (data_train,label_train),(data_valid,label_valid),(data_test,label_test), mapping, scaler
    

def patch_preprocessing(X,sizes=5,kernel_type='gaussian',s=0.1,excluded_idx=[]):
    """Spatial smoothing.
    Parameters
    ----------
    X : array_like
        the 3D hyperspectral image
    size : int
        the size of the smoothing window
    kernel_type : str
        type of kernel to use (e.g. gaussian, triangular, uniform)
    s: float
        the sigma for the Gaussian smoothing kernel
    excluded_idx: array_like
        the indexes of excluded samples
    Returns
    -------
    the dataset and ground-truth arrays which have been deprived
    of samples belonging to undesired classes
    """
    if not isinstance(sizes,list):
        sizes=[sizes]

    if len(excluded_idx)==0:
        excluded_idx=np.full((X.shape[0],X.shape[1]), False, dtype=bool)
    ret=np.zeros((X.shape[0],X.shape[1],X.shape[2]*len(sizes)))
    for nsz,size in enumerate(sizes):
        if kernel_type=='gaussian':
            kernel=gkern(size,s)
        elif kernel_type=='triangular':
            kernel=trkern(size)
        else:
            kernel=np.ones((size,size))
        start=nsz*X.shape[2]
        stop=start+X.shape[2]
        for row in range(X.shape[0]):
            for col in range(X.shape[1]):
                if excluded_idx[row,col]:
                    continue
                n=0
                kern_n=0

                for i in range(size):
                    x=row-(size-1)/2 +i
                    if x<0 or x>=X.shape[0]:
                        continue
                    for j in range(size):
                        y=col-(size-1)/2 +j
                        if y<0 or y>=X.shape[1] or excluded_idx[x,y]:
                            continue

                        if stop>=ret.shape[2]:
                            ret[row,col,start:]=ret[row,col,start:]+(X[x,y,:]*kernel[i,j])
                        else:
                            ret[row,col,start:stop]=ret[row,col,start:stop]+(X[x,y,:]*kernel[i,j])
                        n=n+1
                        kern_n=kern_n+kernel[i,j]


                if stop>=ret.shape[2]:
                    ret[row,col,start:]=ret[row,col,start:]/kern_n
                else:
                    ret[row,col,start:stop]=ret[row,col,start:stop]/kern_n

           
    return ret


def reverseMapping(mapping):
    """Function that produces produces a mapping to unshuffle an array"""
    reverse_mapping=np.zeros(len(mapping),dtype=np.uint64)
    for i in range(len(mapping)):
        reverse_mapping[mapping[i]]=i
    return reverse_mapping


def create2d1dmap(shape):
    """Function that produces a 1D map of 2D object"""
    size=shape[0]*shape[1]
    mapping_2d=np.zeros((size),dtype='uint64,uint64')
    k=0
    for i in range(shape[0]):
        for j in range(shape[1]):
            mapping_2d[k]=(i,j)
            k=k+1
    return mapping_2d


def label_augmentation(y,balance,orig_shape,mapping=[],excluded_mapping_bool=[],mapping_2d=[],window=3,sep_train_test=False,prob=True):
    """Data augmentation adding neighbor pixels of train pixels with the same class
       of the train pixel assigned.
    Parameters
    ----------
    y : array_like
        the original labels
    balance: str
        the percentage of train pixels per class to consider
    mapping: array_like
        the mapping array to unshuffle the dataset
    excluded_mapping_bool: array_like
        the mapping of excluded classes pixels
    mapping_2d: array_like
        same as mapping but 2D shaped
    window: integer
        the size of the neighborhoud
    sep_train_test: boolean
        remove smoothing pixels from the test set
    prob: boolean
        use the probabilistic rule to choose the adequate number of neighbor pixels
    Returns
    -------
    the original dataset plus its perturbed versions
    """
    neigh_window=window if sep_train_test and not window<3 else 3
    if np.asarray(mapping).size>0:
        reverse_mapping=reverseMapping(mapping)

    idxs=balanceSubset(y,balance,mapping_2d,window,True,sep_train_test)
    q,w=np.unique(y[idxs],return_counts=True)
    w=np.asarray(w,dtype=np.float32)
    w_max=np.amax(w)
    w_min=np.amin(w)
    if not sep_train_test:
        if prob:
            max_neigh=np.ones(w.shape,dtype=float)-(w-w_min)/(w_max-w_min)
        else:
            max_neigh=np.round((8/np.log2(100*balance))*(w_max-w)/(w_max-w_min))

    else:
        max_neigh=np.asarray([(neigh_window*neigh_window-1)]*q.size) 

    if np.asarray(mapping).size>0:
        y_orig=y[reverse_mapping]
    else:
        y_orig=y
    if np.asarray(excluded_mapping_bool).size>0:
        excluded_mapping=np.argwhere(excluded_mapping_bool).flatten()
        y_orig=rebuild_from_excluded(y_orig,excluded_mapping,orig_shape[0]*orig_shape[1])

    ret_idxs=[]
    ret_idx_neighbours=[]
    all_selected_neigh=np.zeros(orig_shape,dtype=np.bool)
    all_only_selected_neigh=np.zeros(orig_shape,dtype=np.bool)

    for i in range(idxs.size):
        if np.asarray(excluded_mapping_bool).size>0:
            if np.asarray(mapping).size>0:
                idx=excluded_mapping[mapping[idxs[i]]]
            else:
                idx=excluded_mapping[idxs[i]]
        elif np.asarray(mapping).size>0:
            idx=reverse_mapping[idxs[i]]
        else:
            idx=idxs[i]
        label=y_orig[idx]
        if label<0:
            print('Unexpected error!',idx)
            continue
        sector,shift=divmod(idx,orig_shape[-1])

        selected_neigh=np.zeros(orig_shape,dtype=np.bool)
        only_selected_neigh=np.zeros(orig_shape,dtype=np.bool)
        num_neigh=0
        selected_neigh[sector,shift]=True
        for j in range(neigh_window):
            a=sector-(neigh_window-1)/2 +j
            if a<0 or a>=orig_shape[0]:
                continue
            for k in range(neigh_window):
                b=shift-(neigh_window-1)/2 +k
        
                if k==(neigh_window-1)/2 and j==(neigh_window-1)/2:
                    continue
                if prob:
                    if np.random.rand()>max_neigh[int(label)]:
                        continue
                else:
                    if num_neigh>max_neigh[int(label)]:
                        break
                if b<0 or b>=orig_shape[-1]:
                    continue
                if all_selected_neigh[a,b]:
                    continue
 
                selected_neigh[a,b]=not sep_train_test
                only_selected_neigh[a,b]=True
                num_neigh=num_neigh+1
            if prob:
                if np.random.rand()>max_neigh[int(label)]:
                    continue
            else:
                if num_neigh>max_neigh[int(label)]:
                    break

        y_orig[selected_neigh.flatten()]=label
        all_selected_neigh=np.add(all_selected_neigh,selected_neigh)
        all_only_selected_neigh=np.add(all_only_selected_neigh,only_selected_neigh)

    if np.asarray(mapping).size>0:
        tmp_ret=all_selected_neigh.flatten()[excluded_mapping]
        ret_idxs=np.argwhere(tmp_ret[mapping]).flatten()
        if sep_train_test:
            tmp_ret=all_only_selected_neigh.flatten()[excluded_mapping]
        else:
            tmp_ret=np.logical_and(all_selected_neigh,np.logical_not(all_only_selected_neigh)).flatten()[excluded_mapping]
        ret_idx_neighbours=np.argwhere(tmp_ret[mapping]).flatten()
    else:
        tmp_ret=all_selected_neigh.flatten()[excluded_mapping]
        ret_idxs=np.argwhere(tmp_ret).flatten()
        if sep_train_test:
            tmp_ret=all_only_selected_neigh.flatten()[excluded_mapping]
        else:
            tmp_ret=np.logical_and(all_selected_neigh,np.logical_not(all_only_selected_neigh)).flatten()[excluded_mapping]
        ret_idx_neighbours=np.argwhere(tmp_ret).flatten()

    if np.asarray(excluded_mapping_bool).size>0:
        y_orig=y_orig[excluded_mapping_bool]

    return ret_idxs,y_orig[mapping] if np.asarray(mapping).size>0 else y_orig,ret_idx_neighbours


def balanceSubset(X,n,mapping_2d=[],window=3,categorical_labels=True,separate_mode=False):
    """Build a class balanced train set by return the indices of theano
       sample to use as train set.
    Parameters
    ----------
    X : array_like
        the ground-truth array
    n : int
        the maximum number of samples to use for each class
    Returns
    -------
    an array containing the indices of samples to use as 
    train set
    """
    ret_idx=np.zeros(X.shape,dtype=np.bool)
    black_list_test=np.zeros(X.shape,dtype=np.bool)
    if not categorical_labels:
        raise NotImplementedError('Function not implemented for regression yet')
    classes,n_classes=np.unique(X, return_counts=True)
    if not isinstance(n,list):
        if n<1:
            n=np.ceil(n*n_classes).astype(np.int64).tolist()
            
        else:
            n=[n]*len(classes)
    if any(n_classes[i]<n[i] for i in range(len(n))):
        print(np.unique(X, return_counts=True))
        print('WARNING! Some classes have less than the requested number of samples = ' + str(n[0]))
    tot=0
    tot_n=0
    for c in classes:

        k=0
        tmp=np.in1d(X,c)
        #print(c,n_classes)
        tmp_n=n[int(c)] if n_classes[int(c)]>n[int(c)] else n_classes[int(c)]*2/3
        for i in range(X.shape[0]):
 
            if k<tmp_n:
                ret_idx[i]=ret_idx[i] | tmp[i] 

            else:
                break
            if tmp[i]:
                k=k+1
        tot_n=tot_n+tmp_n
        tot=tot+k

    return np.arange(X.shape[0])[ret_idx]
            

def noise_data_augmentation(data,data_gt,alphas=[0.5,0.5],beta=0.01,factor=2):
    """Data augmentation adding a Random Gaussian Noise to each sample.
    Parameters
    ----------
    data : array_like
        the original dataset
    beta : float
        the constant factor for the Gaussian Noise
    factor : int
        how many perturbed versions to generate
    Returns
    -------
    the original dataset plus its perturbed versions
    """
    if factor<2:
        return data
    augmented_data=np.zeros((factor,data.shape[0],data.shape[-2],data.shape[-1]))
    augmented_data[0,:,:,:]=data[:,:,:]

    data_gt_flat=data_gt.flatten()
    classes,indices=np.unique(data_gt_flat,return_inverse=True)
    size=3
    for i in range(1,factor):

        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                found=False
                for h in range(size):
                    a=x-(size-1)/2 +h
                    for j in range(size):
                        b=y-(size-1)/2 +j
                        if a>=data.shape[0] or b>=data.shape[1] or a<0 or b<0:
                            continue
                        if data_gt[a,b]==data_gt[x,y]:
                            found=True
                            break
                    if found:
                        break
                if not found:
                    raise ValueError("Fellow sample not found in a window of "  +str(size))

                if not data_gt[x,y] == data_gt[a,b]:
                
                    raise ValueError('Fellow sample "',a,b,'"has class "',data_gt[a,b],'" instead of "',data_gt[x,y])
                augmented_data[i,x,y,:]=((alphas[0]*data[x,y,:]+alphas[1]*data[a,b,:])/sum(alphas))
                +beta*np.random.normal(0,1,data.shape[-1])

    return augmented_data   


def load_data(root_dir='./',dataset_name='FTIR',scale_dataset=False,shuffle=-1,balance=0,excluded_class=None,window=1,return_scaler=False,saved_scaler=False, perc_split=[100,0,0]):
    """Function that handles the loading of the considered datasets
    Parameters
    ----------
    root_dir : str
        the parent directory of the dataset directory
    dataset_name : str
        the name of the dataset
    scale_dataset : boolean
        whether to scale or not the train set and the test set according to it
    shuffle : int
        shuffle the dataset according to the input value or not if is equal to -1
    balance: str
        percentage of train samples to retain per class & data augmentation
    excluded_class: array_like
        an arrqay with classes to exclude from the returned sets
    window: str
        size and type of the smoothing window (e.g. gaussian3 is a 3x3 discretized Gaussian filter)
    return_scaler: boolean
        return the scaler used to normalize the train set as output
    Returns
    -------
    X_train : np array
    y_train : np array
    X_test : np array
    y_test : np array
    """
    varname=''

    try:
        if balance.count('l')>0:
            if balance.count('a')>0:
                augmentation_type='all'
            else:
                augmentation_type='label'
        elif balance.count('a')>0:
            augmentation_type='standard'
        else:
            augmentation_type=''
    except:
        augmentation_type=''
        
    augmentation_type = 'label'
    if dataset_name=='PINE':
      if window=='0' or saved_scaler:
        window='gaussian11' 
 
      excluded_class=[0] if excluded_class is None else excluded_class
      varname='indian_pines'
      dataset=root_dir + 'data'+ os.sep + 'Indian_pines'
    elif dataset_name=='SALINA':
      if window=='0' or saved_scaler:
        window='gaussian13'
 
      excluded_class=[0] if excluded_class is None else excluded_class
      varname='salinas'
      dataset=root_dir + 'datasets' + os.sep + 'salinas' + os.sep + 'Salinas'    
    elif dataset_name=='PAVIA':
      if window=='0' or saved_scaler:
        window='gaussian13'

      excluded_class=[0] if excluded_class is None else excluded_class
      varname='pavia'
      dataset=root_dir + 'datasets' + os.sep + 'pavia' + os.sep + 'Pavia'  
    elif dataset_name=='PAVIAU':
      if window=='0' or saved_scaler:
        window='gaussian7'

      excluded_class=[0] if excluded_class is None else excluded_class
      varname='paviaU'
      dataset=root_dir + 'datasets' + os.sep + 'pavia' + os.sep + 'PaviaU'  
    elif dataset_name=='KSC':
      if window=='0' or saved_scaler:
        window='gaussian9'

      excluded_class=[0] if excluded_class is None else excluded_class
      varname='KSC'
      dataset=root_dir + 'datasets' + os.sep + 'ksc' + os.sep + 'KSC' 
    else:
      dataset=root_dir +  os.sep  + dataset_name + '.csv'
    if not os.path.isfile(dataset+'.scaler') and saved_scaler:
        print('Warning! Impossible to retrieve the scaling used! Results might be not those expected!')
        saved_scaler=False
    ######
    train_set, test_set, _, mapping, scaler =fromMAT(path=dataset,varname=varname,perc_split=perc_split, seed=shuffle,scale_dataset=(scale_dataset and not saved_scaler),excluded_class=excluded_class,balance=balance,window=window,augmentation_type=augmentation_type)
    ######
    X_train, y_train = train_set
    y_train_min=np.amin(y_train)
    if y_train.size:
      y_train=y_train-y_train_min
      y_train=y_train.flatten()
    X_test, y_test = test_set
    if y_test.size:
      y_test=y_test-y_train_min
      y_test=y_test.flatten()
   
    if saved_scaler:
        scaler=pickle.load(open(dataset+'.scaler', 'rb'))
        if return_scaler:
            return scaler.transform(X_train), y_train, scaler.transform(X_test), y_test, mapping, scaler
        else:
            return scaler.transform(X_train), y_train, scaler.transform(X_test), y_test, mapping
    else:
        #pickle.dump(scaler,open(dataset+'.scaler', 'wb'))
        if return_scaler:
            return X_train, y_train, X_test, y_test, mapping, scaler
        else:
            return X_train, y_train, X_test, y_test, mapping


def testNetworkInit(net,seed):
  """Function to test the correctness of the random initialization of the weights of the network"""
  are_equals=True
  all_prev_p=[]
  for i in range(10):
    lasagne.random.set_rng(np.random.RandomState(seed))
    net.initialize()
    ls=get_all_layers(net.layers_['output'])
    prev_p=all_prev_p
    all_prev_p=[]
    for l in range(len(ls)):
      l1=ls[l]
      all_param_values = get_all_param_values(l1)
      if i==0:
        all_prev_p.append(all_param_values)
        continue
      for j in range(len(all_param_values)):
        p=all_param_values[j]
        are_equals=np.array_equal(np.asarray(prev_p[l][j]),np.asarray(p))
      all_prev_p.append(all_param_values)
    if not are_equals:
      break
  return are_equals

    
def getCNNParams(filename,path='.' + os.sep + 'saved_networks' + os.sep):
  """ Function to get parameters of a saved neural network"""
  obj={}
  white_list=[
    'update_momentum',
    'conv1d_filter_size',
    'conv1d_num_filters',
    'conv1d_stride',
#    'gaussian_sigma',
    'objective_lamda1',
    'objective_lamda2',
#    'seed',
    
  ]
  # The '-' character is to indicate the name of the property in the second object
  getvalue_list=[
    'on_epoch_finished-start'
  ]
  rename_getvalue_list=[
    'learning_rate'
  ]
  net=loadVar(filename,path)
  
  for attr in dir(net):
    if not attr in white_list and not any(attr in s for s in getvalue_list):
      continue
    if any(attr in s for s in getvalue_list):
      tmp_attrs=[s for s in getvalue_list if attr in s]
      if not len(tmp_attrs)==1:
        print('Sub attributes found > 1!! Not implemented',tmp_attrs,'\n SKIPPING')
        break
          
      white_list2=[tmp_attrs[0].split('-')[0]]
      sub_attrs=tmp_attrs[0].split('-')[1:]
      #print(white_list2,sub_attrs)
      tmp_val=getattr(net,white_list2[0])
      for i in tmp_val:
        for attr2 in dir(i):
          if not attr2 in sub_attrs:
            continue
          if len(rename_getvalue_list)>getvalue_list.index(attr+'-'+attr2):
            obj[rename_getvalue_list[getvalue_list.index(attr+'-'+attr2)]]=getattr(i,attr2)
          else:
            obj[attr+'-'+attr2]=getattr(i,attr2)
    else:
      #print(attr,getattr(net,attr))
      obj[attr]=getattr(net,attr)
  return obj


def printNet(net):
  """Simply dump to stdout a netowrk object"""
  print(net)


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def trkern(size=7):
    """Returns a 2D triangular kernel array."""
    first_row=np.ones(size)
    kernel=np.zeros((size,size))
    for i in range(1,(size+1)/2):
        first_row[i]=first_row[i-1]+1
        first_row[-i-1]=first_row[-i]+1
    for i in range((size+1)/2):  
        kernel[i,:]=(i+1)*first_row
        kernel[-i-1,:]=(i+1)*first_row
    return kernel


def sigmoid(x,beta=1.0,alpha=0.0):
    a = []
    for item in x:
        a.append((2.0/(1+exp(-beta*(item+alpha))))-1.0)
    return np.array(a,dtype=np.float64)
