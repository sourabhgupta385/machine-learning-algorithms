import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Linear Regression/ex1data2.txt', header = None) #read from dataset
X = data.iloc[:,0:2] # read first two column
y = data.iloc[:,2] # read third column
m = len(y) # number of training example

X = (X - np.mean(X)) / np.std(X) # feature normalization
y = y[:,np.newaxis]
theta = np.zeros((3,1))
iterations = 400
alpha = 0.01
ones = np.ones((m,1))
X = np.hstack((ones, X)) # adding the intercept term

def computeCost(X, y, theta):
    temp = np.dot(X, theta) - y
    return np.sum(np.power(temp, 2)) / (2*m)

J = computeCost(X, y, theta)
print(J)

def gradientDescent(X, y, theta, alpha, iterations):
    for _ in range(iterations):
        temp = np.dot(X, theta) - y
        temp = np.dot(X.T, temp)
        theta = theta - (alpha/m) * temp
    return theta
theta = gradientDescent(X, y, theta, alpha, iterations)
print(theta)

J = computeCost(X, y, theta)
print(J)
