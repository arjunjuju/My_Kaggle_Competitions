# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 00:14:36 2017

@author: Arjun
"""
#implementing machine learning algorithms for Kaggle-titanic data ste
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#we are going to perform three predictions using three regression based models and select best one
titanic_training_file = 'C:/Users/Arjun/Desktop/Project december/Kaggle/train.csv';
titanic_testing_file = 'C:/Users/Arjun/Desktop/Project december/Kaggle/test.csv';
titanic_contents = pd.read_csv(titanic_training_file);#loading the file and reading the contents
print(titanic_contents);
titanic_test_contents = pd.read_csv(titanic_testing_file);
 #selecting the columns for pedicting
#titanic_contents["Age"].fillna(value=titanic_contents["Age"].median());
 #modifying and filling up missing age values using alternative approaches
 #instead of just median values
 #training data set
average_age_titanic   = titanic_contents["Age"].mean();# Mean
std_age_titanic       = titanic_contents["Age"].std();# Standard Deviation
count_nan_age_titanic = titanic_contents["Age"].isnull().sum();#count of empty values
#testing data set
average_age_test   = titanic_test_contents["Age"].mean();# Mean
std_age_test       = titanic_test_contents["Age"].std();# Standard deviation
count_nan_age_test = titanic_test_contents["Age"].isnull().sum();#count of empty values
#generating random numbers
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic);
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test);
#dropping null values and converting to int
titanic_contents['Age'].dropna().astype(int);
titanic_test_contents['Age'].dropna().astype(int);
#filling empty values with random ones
titanic_contents["Age"][np.isnan(titanic_contents["Age"])] = rand_1;
titanic_test_contents["Age"][np.isnan(titanic_test_contents["Age"])] = rand_2;
#converting data type of Age
titanic_contents['Age'] = titanic_contents['Age'].astype(int);
titanic_test_contents['Age'] = titanic_test_contents['Age'].astype(int);
#titanic_test_contents["Age"].fillna(value=titanic_test_contents["Age"].median()); 
titanic_predictors = ['PassengerId','Pclass','Age','SibSp','Parch'];
titanic_X_train = titanic_contents[titanic_predictors].fillna(value=titanic_contents["Age"].median());
titanic_y_train = titanic_contents.Survived;
titanic_X_test = titanic_test_contents[titanic_predictors].fillna(value=titanic_test_contents["Age"].median()); 
print(titanic_X_train);
print(titanic_X_test);
# =============================================================================
# knn = KNeighborsClassifier(n_neighbors = 3);
# knn.fit(titanic_X_train,titanic_y_train);
# knn_pred = knn.predict(titanic_X_test);
# print(knn.score(titanic_X_train,titanic_y_train));   
# submission = pd.DataFrame({
#         "PassengerId": titanic_X_test["PassengerId"],
#         "Survived": knn_pred
#     })
# submission.to_csv('titanicknn1.csv', index=False)
# =============================================================================
# =============================================================================
# Logreg = LogisticRegression();
# Logreg.fit(titanic_X_train,titanic_y_train);
# logreg_pred = Logreg.predict(titanic_X_test);
# print(Logreg.score(titanic_X_train,titanic_y_train));   
# submission = pd.DataFrame({
#         "PassengerId": titanic_X_test["PassengerId"],
#         "Survived": logreg_pred
#         })
# submission.to_csv('titaniclogreg.csv', index=False);
# 
# =============================================================================
ranfor = RandomForestClassifier();
ranfor.fit(titanic_X_train,titanic_y_train);
ranfor_pred = ranfor.predict(titanic_X_test);
print(ranfor.score(titanic_X_train,titanic_y_train));   
submission = pd.DataFrame({
        "PassengerId": titanic_X_test["PassengerId"],
        "Survived": ranfor_pred
        })
submission.to_csv('titanicranfor.csv', index=False);

# Support Vector Machines
svc = SVC();
svc.fit(titanic_X_train,titanic_y_train);
svc_pred = svc.predict(titanic_X_test)
print(svc.score(titanic_X_train,titanic_y_train));
submission = pd.DataFrame({
        "PassengerId": titanic_X_test["PassengerId"],
        "Survived": svc_pred
        })
submission.to_csv('titanicsvc.csv', index=False);

