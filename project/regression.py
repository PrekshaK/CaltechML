import numpy as np 
import scipy as sp 
import sklearn as sk
from sklearn import tree
from sklearn.linear_model import LogisticRegression


#importing dataset
dataset= np.genfromtxt('parole.csv', delimiter=',')

#importing test dataset
testdataset = np.genfromtxt('paroletest.csv', delimiter=",")

#x and y dataset for test
y = testdataset[:, -1]
x = testdataset[:, [0, -3]]

#x and y for training
X = dataset[:,[0, -3]]
Y = dataset[:, -1]

#printing y to see the required prediction
print y

#Using decision tree to predict the data
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)  #Fitting data or training given dataset
print clf.predict(x)

#Using logistic regression to predict the data
logreg = LogisticRegression()
logreg.fit(X, Y) #Fitting data or training given dataset
predicted =  logreg.predict(x)

#Printing the predicted data
print predicted

#Assigning expected to the y of testdataset
expected = y

#Checking the accuracy of prediction
summarize = sk.metrics.classification_report(expected, predicted)

#Printing the result
print summarize

