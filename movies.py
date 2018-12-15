# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv('movies_dataset.csv')
used_features = ["director", "starring","type","rate","votes","region","runtime","language","company"]
print(len(dataset))

#print(dataset[used_features].duplicated())
dataset=dataset[used_features].drop_duplicates()
print(len(dataset))

dataset=dataset.applymap(lambda x: x.replace('\'','').replace(r"\n","").strip()).applymap(lambda x: np.NaN if str(x).isspace() or x=='null' else x)

#print(dataset.isnull())
dataset=dataset.dropna()
print(len(dataset))

number_features = ['rate','votes','runtime']
dataset[number_features]=dataset[number_features].applymap(lambda x: float(x) if x.isdigit() else -1)
dataset = dataset[dataset['votes']>0]
dataset = dataset[dataset['rate']>0]
dataset = dataset[dataset['runtime']>0]
print(len(dataset))

X = dataset[used_features]
Y = dataset["rate"]

labelencoder = LabelEncoder()
X["director"] = labelencoder.fit_transform(X["director"])
X["company"] = labelencoder.fit_transform(X["company"])
X["starring"] = labelencoder.fit_transform(X["starring"])
X["type"] = labelencoder.fit_transform(X["type"])
X["region"] = labelencoder.fit_transform(X["region"])
X["language"] = labelencoder.fit_transform(X["language"])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state=0)

# 线形回归
lr = LinearRegression()
lr.fit(X_train, y_train)
y_predict = lr.predict(X_test)
print(y_predict)
print(lr.score(X_test, y_test))

# 决策树
dtr=DecisionTreeRegressor()
dtr.fit(X_train,y_train)
y_predict = dtr.predict(X_test)
print(dtr.score(X_test, y_test))
