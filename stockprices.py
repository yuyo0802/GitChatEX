# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 22:10:46 2018

@author: Administrator
"""

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

"""
LinearRegression
"""
# Importing dataset
data = pd.read_csv('msft_stockprices_dataset.csv', delimiter=',')        
used_features = ["High Price", "Low Price","Open Price","Volume"]
X = data[used_features].values
close_price = data["Close Price"].values

# 从数据集中取20%作为测试集，其他作为训练集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    close_price,
    test_size=0.2,
    random_state=0,
)

# 线形回归
regr = LinearRegression()
regr.fit(X_train, y_train)
y_predict = regr.predict(X_test)
print(regr.score(X_test, y_test))

# 决策树
regr=DecisionTreeRegressor()
regr.fit(X_train,y_train)
y_predict = regr.predict(X_test)
print(regr.score(X_test, y_test))