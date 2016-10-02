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

    # normalize data
    # x = a1.normalize_data(x)

    # set param value
    PlotMatrixRMSErorr, PlotMatrixRMSErorrTest = calculateToPlot(x, targets)
    # Set EW
    # Ew = t_train - (0.5*np.square(PhiTrain))*w
    plotMatrixDegree = np.matrix([[1], [2], [3], [4], [5], [6]])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotMatrixDegree, PlotMatrixRMSErorr)
    plt.ylabel('RMS')
    plt.legend(['Test error', 'Training error'])
    plt.title('Fit with polynomials, no regularization')
    plt.xlabel('Polynomial degree')
    plt.subplot(211)
    plt.plot(plotMatrixDegree, PlotMatrixRMSErorrTest)
    plt.ylabel('RMS')
    plt.legend(['Test error', 'Training error'])
    plt.title('Fit with polynomials, no regularization')
    plt.xlabel('Polynomial degree')

    x = a1.normalize_data(x)
    PlotMatrixRMSErorr, PlotMatrixRMSErorrTest = calculateToPlot(x, targets)

    plt.subplot(212)
    plt.plot(plotMatrixDegree, PlotMatrixRMSErorr)
    plt.ylabel('RMS')
    plt.legend(['Test error', 'Training error'])
    plt.title('Fit with polynomials, no regularization and normalize')
    plt.xlabel('Polynomial degree')
    plt.subplot(212)
    plt.plot(plotMatrixDegree, PlotMatrixRMSErorrTest)
    plt.ylabel('RMS')
    plt.legend(['Test error', 'Training error'])
    plt.xlabel('Polynomial degree')

    plt.show()


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


def calculateToPlot(x, targets):
    N_TRAIN = 100;
    x_train = x[0:N_TRAIN, :]
    x_test = x[N_TRAIN:, :]
    t_train = targets[0:N_TRAIN]
    t_test = targets[N_TRAIN:]

    PlotMatrixRMSErorr = np.zeros((6, 1))
    PlotMatrixRMSErorrTest = np.zeros((6, 1))
    for i in range(1, 7):
        PhiTrain = calculateThetaTrainPolynomial(x_train, N_TRAIN, i)
        PhiTest = calculateThetaTrainPolynomial(x_test, 95, i)
        # Set W
        sInverce = np.linalg.pinv(PhiTrain)
        w = sInverce * t_train
        ourTrainValidation = PhiTrain * w
        ourTestValidation = PhiTest * w
        PlotMatrixRMSErorr[i - 1] = RMS(t_train, ourTrainValidation)
        PlotMatrixRMSErorrTest[i - 1] = RMS(t_test, ourTestValidation)
    return PlotMatrixRMSErorr, PlotMatrixRMSErorrTest


main()
