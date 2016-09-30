#!/usr/bin/env python

import assignment1 as a1
import numpy as np

from numpy.linalg import inv
from numpy import transpose
import matplotlib.pyplot as plt

#Get input data
(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]

#normalize data
#x = a1.normalize_data(x)

#set param value
N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

#set Matrix ThetaTrain
ThetaTrain = np.ones((N_TRAIN,1), dtype=np.int)
ThetaTrain = np.concatenate((ThetaTrain,x_train),axis=1)

#set Matrix ThetaTest
ThetaTest = np.ones((N_TRAIN,1), dtype=np.int)
ThetaTest = np.concatenate((ThetaTest,t_train),axis=1)

#multThetaAndThetaT
ThetaMul = np.transpose(ThetaTrain)*(ThetaTrain)
#Inverse
inveThetaMul = inv(ThetaMul)
#set right of w
rightW = np.transpose(ThetaTrain)*(t_train)

#Set W
w = inveThetaMul*rightW
print w.shape
print ThetaTrain.shape
print transpose(w).shape
print ThetaTrain*w
Ew = t_train
print Ew.shape

