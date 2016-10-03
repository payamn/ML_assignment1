#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv


def main():
    # Get input data
    (countries, features, values) = a1.load_unicef_data()

    targets = values[:, 1]

    x = values[:, 7:]
    x = a1.normalize_data(x)
    x = x[0:100, :]
    targets = targets[0:100,:]
    # normalize data
    #
    minAvgRMS = 999999999
    landas = [0,0.01,0.1,1,10,100,1000,10000]
    avgErrorForEachL = []
    for landa in landas:
        start = 0
        step = 10
        avgErrorRMS = 0
        for i in range (0,10):
            x_train, t_train, x_validation, t_validation  = setXes(i*10, x, targets)
            PlotMatrixRMSErorr = crossValidationCal(x_train, t_train, landa, x_validation, t_validation)
            avgErrorRMS += PlotMatrixRMSErorr[0,0]
        avgErrorRMS = avgErrorRMS/10
        print avgErrorRMS , landa
        avgErrorForEachL.append(avgErrorRMS)
    plt.semilogx(landas,avgErrorForEachL)

    plt.show()




def setXes(index, x, targets):

    x_train = np.r_[x[0:index,:], x[index+10:,:]]
    t_train = np.r_[targets[0:index,:], targets[index + 10:,:]]

    t_validation = targets[index:index+10,:]
    x_validation = x[index:index + 10,:]
    return x_train, t_train, x_validation, t_validation

def RMSLanda(matrixA, matrixB,w):
    temp = matrixA - matrixB
    temp = np.square(temp)
    sumTemp = sum(temp)
    sumTemp = sumTemp / 10
    sumTemp = np.sqrt(sumTemp)
 #   print np.square(w)

    return sumTemp


# X is all the value, N is
def calculateThetaTrainPolynomial(X, N, polynomialDegree):
    ThetaTrain = np.ones((N, 1), dtype=np.int)
    for i in range(1, polynomialDegree + 1):
        tempX = np.power(X, i)
        #    print X
        ThetaTrain = np.c_[ThetaTrain, tempX]
        # ThetaTrain = np.concatenate((ThetaTrain,X),axis=1)
    return ThetaTrain


def crossValidationCal(x_train, t_train,landa, x_validation, t_validation ):

    PhiTrain = calculateThetaTrainPolynomial(x_train, 90, 2)
    PhiValidation = calculateThetaTrainPolynomial(x_validation,10 ,2)
    # Set W
    sInverce = np.linalg.inv (landa*np.eye(PhiTrain.shape[1])+ np.transpose(PhiTrain)*PhiTrain) * np.transpose(PhiTrain)
    w = sInverce * t_train
    ourTrainValidation = PhiValidation * w
    PlotMatrixRMSErorr = RMSLanda(t_validation, ourTrainValidation, w)
    return PlotMatrixRMSErorr


main()
