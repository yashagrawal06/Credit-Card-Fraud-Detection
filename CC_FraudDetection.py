# -*- coding: utf-8 -*-

#Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Importing the dataset
data = pd.read_csv("creditcard.csv")
data = pd.DataFrame(data)

#Creating a copy of the dataset
dataset = data

#Inspecting the data
print(data.head())
print(data.shape)
print(list(data.columns))
data.dtypes

#Check for Null Values
data.isnull().sum()

#Inspecting proportion of classes
sns.countplot( x= 'Class', data = data)
data['Class'].value_counts()

#Plots for Amount and Time
sns.distplot(data['Amount'])
sns.distplot(data['Time'])
plt.boxplot(data['Amount'])

#Scaling Amount using Robust Scaler
data['sc_amount'] = RobustScaler().fit_transform(data['Amount'].values.reshape(-1,1))

#Dropping Time variable
data = data.drop(['Time'],axis=1)

#Dropping original Amount column
data = data.drop(['Amount'], axis = 1)
data.info()

#Creating Train and Test dataset
X = data.iloc[:, data.columns != 'Class']
Y = data.iloc[:, data.columns == 'Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 25)
X_train.shape
X_test.shape

#Using SMOTE to handle the imbalaned dataset
sm_sampling = SMOTE(random_state=26)
sm_X_train, sm_Y_train = sm_sampling.fit_sample(X_train, Y_train)

#Logistic Regression
lr_classifier = LogisticRegression(random_state = 0)
lr_classifier.fit(sm_X_train, sm_Y_train)

#Predicting Test Set results
y_pred = lr_classifier.predict(X_test)

#Confusion matrix for Logistic Refression Classifier
cm_lr = confusion_matrix(Y_test, y_pred)
print(cm_lr)

#Other evaluation metrics(Recall, F1-Score, Precision)
print(classification_report(Y_test, y_pred))