import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

data = pd.read_csv('logistic regression/ex2data1.txt', header = None) #read from dataset
X = data.iloc[:,0:2] # read first column
y = data.iloc[:,2] # read second column

(m, n) = X.shape
X = np.hstack((np.ones((m,1)), X))
y = y[:, np.newaxis]
theta = np.zeros((n+1,1)) # intializing theta with all zeros

def sigmoid(x):
  return 1/(1+np.exp(-x))

def costFunction(theta, X, y):
    J = (-1/m) * np.sum(np.multiply(y, np.log(sigmoid(np.dot(X,theta)))) 
        + np.multiply((1-y), np.log(1 - sigmoid(np.dot(X,theta)))))
    return J

def gradient(theta, X, y):
    return ((1/m) * X.T @ (sigmoid(np.dot(X,theta)) - y))  

J = costFunction(theta, X, y)
print(J)

temp = opt.fmin_tnc(func = costFunction, 
                    x0 = theta.flatten(),fprime = gradient, 
                    args = (X, y.flatten()))

theta_optimized = temp[0]
print(theta_optimized)

J = costFunction(theta_optimized[:,np.newaxis], X, y)
print(J)

plot_x = [np.min(X[:,1]-2), np.max(X[:,2]+2)]
plot_y = -1/theta_optimized[2]*(theta_optimized[0] 
          + np.dot(theta_optimized[1],plot_x))  

mask = y.flatten() == 1
adm = plt.scatter(X[mask][:,1], X[mask][:,2],label='Admitted')
not_adm = plt.scatter(X[~mask][:,1], X[~mask][:,2],label='Not Admitted')
decision_boun = plt.plot(plot_x, plot_y, label='Decision Boundary')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()

def accuracy(X, y, theta, cutoff):
    pred = [sigmoid(np.dot(X, theta)) >= cutoff]
    acc = np.mean(pred == y)
    print(acc * 100)

accuracy(X, y.flatten(), theta_optimized, 0.5)
