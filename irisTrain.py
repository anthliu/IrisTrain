#!/usr/bin/env python

import numpy as np

def sigmoid(inputVar, parameters):
    return 1 / ( 1 + np.exp( -1 * np.dot(inputVar, parameters) ))

def preprocess(inputVar):
    return np.insert(inputVar, 0, 1, axis = 1) # change the input variables to make computation of theta[0] easier

def subset(trainingTarg, tag): #for one vs. all training
    subTarg = []
    for x in trainingTarg:
        if x == tag:
            x = 1
        else:
            x = 0
        subTarg.append([x])
    return np.array(subTarg)

def cost(inputVar, trainingTarg, parameters, regPar):
    exNumb = trainingTarg.shape[0]
    regPart = (regPar / (2 * exNumb)) * (parameters ** 2).sum()
    prediction = sigmoid(inputVar, parameters)
    oPart = (1./exNumb) * ( - trainingTarg * np.log(prediction) - ( 1. - trainingTarg ) * np.log( 1. - prediction )).sum()
    return regPart + oPart

def batchGradDescent(inputVar, trainingTarg, rate, regPar, errorMarg = 0.001):
    parameters = np.random.rand(inputVar.shape[1] + 1, 1) * 10 - 5
    convergence = False
    inputVar = preprocess(inputVar)
    avgMod = trainingTarg.shape[0]
    while convergence == False:
        parametersOld = np.array(parameters)
        #update parameters:
        for j in range(0, parametersOld.size):
            derivative = ( (sigmoid(inputVar, parameters) - trainingTarg) * inputVar[:, j].T ).sum() + regPar * parameters[j, 0]
            parameters[j, 0] = parametersOld[j, 0] - (rate / avgMod) * derivative
        #test for convergence:
        convergence = True
        for j in range(0, parametersOld.size):
            if np.absolute(1 - (parametersOld[j] / parameters[j])) > errorMarg:
                convergence = False #if any parameter change is greater than the error margin, it will switch back to false
    return parameters

def predict(inputVar, parameters):
    return sigmoid(preprocess(inputVar), parameters)
    
def preprocessTarget(trainingTarg):
    target = []
    for x in trainingTarg:
        if x[0] == "Iris-setosa":
            x = 0
        elif x[0] == "Iris-versicolor":
            x = 1
        else: # Iris-virginica
            x = 2
        target.append([x])
    return np.array(target)

def loadIris():
    inputVar = []
    trainingTarg = []
    with open('iris.data', 'r') as fstream:
        for line in fstream:
            line = line.split(',')
            subline = []
            for i in line[:-1]:
                subline.append(float(i))
            inputVar.append(subline)
            trainingTarg.append([line[-1].strip()])
    trainingTarg = preprocessTarget(trainingTarg)
    return (np.array(inputVar), np.array(trainingTarg))

def main():
    inputVar, trainingTarg = loadIris()

    #shuffle
    joined = np.hstack((inputVar, trainingTarg))
    np.random.shuffle(joined)
    inputVar = joined[:,:-1]
    trainingTarg = joined[:,-1][np.newaxis].T

    exampleNumb = 120

    Tinput = inputVar[:exampleNumb]
    Ttarg = trainingTarg[:exampleNumb]

    Pinput = inputVar[exampleNumb:]
    Ptarg = trainingTarg[exampleNumb:]

    rate = 0.001
    reg = 0
    errorMarg = .00001
    setpar = batchGradDescent(Tinput, subset(Ttarg, 0), rate, reg, errorMarg)
    verspar = batchGradDescent(Tinput, subset(Ttarg, 1), rate, reg, errorMarg)
    virgpar = batchGradDescent(Tinput, subset(Ttarg, 2), rate, reg, errorMarg)

    print setpar
    print verspar
    print virgpar
    
    correct = 0.
    incorrect = 0.

    setpred = predict(Pinput, setpar)
    verspred = predict(Pinput, verspar)
    virgpred = predict(Pinput, virgpar)

    for x in range(0, len(setpred)):
        prediction = np.argmax( (setpred[x], verspred[x], virgpred[x]) )
        if  prediction == Ptarg[x]:
            correct += 1
        else:
            incorrect += 1
    print "Percentage correct: ", 100 * (correct / (correct + incorrect))

if __name__ == "__main__":
    main()
