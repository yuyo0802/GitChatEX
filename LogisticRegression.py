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

# Linear Regression - Regression
y_train = scores[:11]
y_test = scores[11:]

regr = LinearRegression()
regr.fit(X_train, y_train)
y_predict = regr.predict(X_test)

# 对比测试结果和预期结果
print(regr.score(X_test, y_test))

