# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.svm import SVC

dataset = pd.read_csv('employees_dataset.csv')
X = dataset.iloc[ : , :2].values
Y = dataset.iloc[ : , 4].values

labelencoder = LabelEncoder()
X[ : , 0] = labelencoder.fit_transform(X[ : , 0])
X[ : ,1] = labelencoder.fit_transform(X[ : , 1])

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9)
vectorizer.fit(dataset['skills'])
X=np.concatenate((X,vectorizer.transform(dataset['skills']).todense()), axis=1)

vectorizer.fit(dataset['working_experience'])
X=np.concatenate((X,vectorizer.transform(dataset['working_experience']).todense()), axis=1)

Y =  labelencoder.fit_transform(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)

gnb = MultinomialNB()
gnb.fit(X_train, y_train)
y_predict = gnb.predict(X_test)
print(gnb.score(X_test, y_test))

classifier = LogisticRegression(C=1e5)
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
print(classifier.score(X_test, y_test))

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
print(clf.score(X_test, y_test))

model = SVC(kernel='linear', C=1.0)
#model = SVC(kernel='rbf', C=50)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(model.score(X_test, y_test))


