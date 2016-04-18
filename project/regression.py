import numpy as np 
import scipy as sp 
import sklearn as sk
from sklearn import tree
from sklearn.linear_model import LogisticRegression

import os

dataset= np.genfromtxt('parole.csv', delimiter=',')
testdataset = np.genfromtxt('paroletest.csv', delimiter=",")

y = testdataset[:, -1]
x = testdataset[:, [0, -3]]

X = dataset[:,[0, -3]]
Y = dataset[:, -1]
print y

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
print clf.predict(x)

logreg = LogisticRegression()

logreg.fit(X, Y)
predicted =  logreg.predict(x)
print predicted

expected = y

summarize = sk.metrics.classification_report(expected, predicted)
print summarize


#71.45
