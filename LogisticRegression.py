# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 21:26:55 2018

@author: Administrator
"""

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import pandas as pd

"""
LinearRegression
"""
# Importing dataset
data = pd.read_csv('quiz.csv', delimiter=',')        
used_features = ["Last Score", "Hours Spent"]
X = data[used_features].values
scores = data["Score"].values

X_train = X[:11]
X_test = X[11:]

"""
# Linear Regression - Regression
y_train = scores[:11]
y_test = scores[11:]

regr = LinearRegression()
regr.fit(X_train, y_train)
y_predict = regr.predict(X_test)

print(y_predict)
"""
"""
# Logistic Regression – Binary Classification
passed = []

for i in range(len(scores)):
    if(scores[i] >= 60):
        passed.append(1)
    else:
        passed.append(0)

y_train = passed[:11]
y_test = passed[11:]
"""

# Logistic Regression - Multiple Classification
level = []

for i in range(len(scores)):
    if(scores[i] >= 85):
        level.append(2)
    elif(scores[i] >= 60):
        level.append(1)
    else:
        level.append(0)

y_train = level[:11]
y_test = level[11:]


classifier = LogisticRegression(C=1e5)
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)
print(y_predict)

