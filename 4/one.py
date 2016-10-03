#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
#x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

index1990 =  np.argmax(values[:,0])
print "Highest child mortality rate in 1990: "+ countries[index1990] +" rate: "+ str(values[index1990,0])

index2011 = np.argmax(values[:,1])
print "highest child mortality rate in 2011: "+ countries[index2011] + " rate: "+ str(values[index2011,1])

