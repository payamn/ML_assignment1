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
    #x = values[:, 7:]

    # normalize data
    # x = a1.normalize_data(x)

    # set param value
    PlotMatrixRMSErorr =[]
    PlotMatrixRMSErorrTest = []
    for i in range (7 , 15):
        x = values[:,i]
        rmsErrorTrain, rmsErorrTest = calculateToPlotPolynomial3(x, targets)
        PlotMatrixRMSErorr.append(rmsErrorTrain[0,0])
        PlotMatrixRMSErorrTest.append(rmsErorrTest[0,0])


    # Set EW
    # Ew = t_train - (0.5*np.square(PhiTrain))*w
    plotMatrixDegree = np.matrix([[1], [2], [3], [4], [5], [6],[7],[8]])
    labels = features [7:15]
    plt.xticks([1,2,3,4,5,6,7,8],labels,rotation='vertical')
    plt.figure(1)
    plt.subplot(111)
    plt.tight_layout()
    #plt.plot(plotMatrixDegree, PlotMatrixRMSErorr)
    plt.bar(plotMatrixDegree-0.2, PlotMatrixRMSErorr , width=0.1, color="red")
    plt.ylabel('RMS')
    plt.legend([ 'Training error', 'Test error'])
    plt.title('Feature 8 to 16')
    #plt.xlabel('Polynomial degree')

    plt.subplot(111)
    # plt.plot(plotMatrixDegree, PlotMatrixRMSErorr)
    plt.bar(plotMatrixDegree, PlotMatrixRMSErorrTest, width=0.1, color="Blue")
    plt.ylabel('RMS')
    plt.legend(['Training error', 'Test error'])
    plt.title('Feature 8 to 16')
    #plt.xlabel('Polynomial degree')

    plt.figure(2)
    for i in range (0,3):
        plt.subplot(311+i)
        x_train, t_train, ourTrainValidation, x_test, t_test, wml= getFitPolynomial3(values[:,10+i],targets)

        t1 = np.arange(min(x_test[:,0]), max(x_test[:,0]), 0.1)
        plt.plot(t1, calCurve(t1,wml),color = "Green")
        plt.plot(x_train, t_train, 'ro', color="Blue")
        #plt.plot(x_train, ourTrainValidation, 'k', color="Red")
        plt.plot(x_test, t_test, 'ro', color="Yellow")
        plt.legend(['Learned polynomial','Train point', 'Test Point'],bbox_to_anchor=(1.00, 0.73), loc=1, borderaxespad=0)

    plt.show()

def calCurve(myX,wml):
    return wml[0,0] + wml[1,0] * myX + wml[2,0] * myX * myX + wml[3,0] * myX * myX * myX

def RMS(matrixA, matrixB):
    temp = matrixA - matrixB
    temp = np.square(temp)
    sumTemp = sum(temp)
    sumTemp = sumTemp / 100
    sumTemp = np.sqrt(sumTemp)

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


def calculateToPlotPolynomial3(x, targets):
    N_TRAIN = 100;
    x_train = x[0:N_TRAIN, :]
    x_test = x[N_TRAIN:, :]
    t_train = targets[0:N_TRAIN]
    t_test = targets[N_TRAIN:]

    i = 3
    PhiTrain = calculateThetaTrainPolynomial(x_train, N_TRAIN, i)
    PhiTest = calculateThetaTrainPolynomial(x_test, 95, i)
    # Set W
    sInverce = np.linalg.pinv(PhiTrain)
    w = sInverce * t_train
    ourTrainValidation = PhiTrain * w
    ourTestValidation = PhiTest * w
    PlotMatrixRMSErorr = RMS(t_train, ourTrainValidation)
    PlotMatrixRMSErorrTest = RMS(t_test, ourTestValidation)
    return PlotMatrixRMSErorr, PlotMatrixRMSErorrTest


def getFitPolynomial3(x, targets):
    N_TRAIN = 100;
    x_train = x[0:N_TRAIN, :]
    x_test = x[N_TRAIN:, :]
    t_train = targets[0:N_TRAIN]
    t_test = targets[N_TRAIN:]

    i = 3
    PhiTrain = calculateThetaTrainPolynomial(x_train, N_TRAIN, i)
    PhiTest = calculateThetaTrainPolynomial(x_test, 95, i)
    # Set W
    sInverce = np.linalg.pinv(PhiTrain)
    w = sInverce * t_train
    ourTrainValidation = PhiTrain * w
    ourTestValidation= PhiTest * w
    return x_train,t_train, ourTrainValidation, x_test, t_test,w
main()
