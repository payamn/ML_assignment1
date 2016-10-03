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
    x = values[:, 10]
    mue = 100
    subplot = 211
    for mue in [100,10000]:
        plt.figure(1)
        plt.subplot(subplot)
        PlotMatrixRMSErorr, PlotMatrixRMSErorrTest, x_train, t_train, ourTrainValidation, x_test, t_test, wml= calculateToPlot(x,targets,mue)
        print PlotMatrixRMSErorr, PlotMatrixRMSErorrTest
        t1 = np.arange(min(x_test[:,0]), max(x_test[:,0]), 0.1)
        plt.plot(t1, calCurve(t1,wml,mue),color = "Green")
        plt.plot(x_train, t_train, 'ro', color="Blue")
        #plt.plot(x_train, ourTrainValidation, 'k', color="Red")
        plt.plot(x_test, t_test, 'ro', color="Yellow")
        plt.legend(['Learned polynomial','Train point', 'Test Point'],bbox_to_anchor=(1.00, 0.73), loc=1, borderaxespad=0)


        subplot = subplot + 1
        #PlotMatrixRMSErorr, PlotMatrixRMSErorrTest = calculateToPlot(x, targets)
        # Set EW
        # Ew = t_train - (0.5*np.square(PhiTrain))*w
    plt.show()
def calCurve(x,w,mue):
    return w[0,0]+w[1,0]* (1/((1+np.exp((mue-x)/2000.0))))

def RMS(matrixA, matrixB):
    temp = matrixA - matrixB
    temp = np.square(temp)
    sumTemp = sum(temp)
    sumTemp = sumTemp / 100
    sumTemp = np.sqrt(sumTemp)
    return sumTemp


# X is all the value, N is
def calculatePhiTrainSigmoid(X, N,mue):
    phiTrain = np.ones((X.shape[0], X.shape[1]+1), dtype=np.float)
    for i in range(0, X.shape[0]):
        for j in range (0,X.shape[1]):
            phiTrain[i,j+1] = 1/((1+math.exp((mue-X[i,j])/2000.0)))
    return phiTrain

def calculateToPlot(x, targets,mue):
    N_TRAIN = 100;
    x_train = x[0:N_TRAIN, :]
    x_test = x[N_TRAIN:, :]
    t_train = targets[0:N_TRAIN]
    t_test = targets[N_TRAIN:]

    PhiTrain = calculatePhiTrainSigmoid(x_train, N_TRAIN, mue)
    PhiTest = calculatePhiTrainSigmoid(x_test, 95, 100)
        # Set W
    sInverce = np.linalg.pinv(PhiTrain)
    w = sInverce * t_train
    ourTrainValidation = PhiTrain * w
    ourTestValidation = PhiTest * w
    PlotMatrixRMSErorr = RMS(t_train, ourTrainValidation)
    PlotMatrixRMSErorrTest = RMS(t_test, ourTestValidation)
    return PlotMatrixRMSErorr, PlotMatrixRMSErorrTest, x_train, t_train, ourTrainValidation, x_test, t_test, w


main()
