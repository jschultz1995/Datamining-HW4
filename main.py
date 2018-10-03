# -*- coding: utf-8 -*-
"""
Spyder Editor
Josh Schultz
This is my program to compare the KNN and Decision tree classifiers using the iris data set
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn import preprocessing
import seaborn as sns

def get_acc_for_k(X_train, X_test, Y_train, Y_test, k):
    clf = neighbors.KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    return ([k, accuracy_score(Y_test, y_pred)])

#Getting the Data Set and splitting it
full_df = pd.read_csv("iris.csv")
target = full_df.values[:,4]
predictors = full_df.values[:,0:4]
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size = 0.2, random_state = 100)

#Decision Tree Routine
my_tree = DecisionTreeClassifier()
my_tree.fit(X_train, Y_train)

y_pred = my_tree.predict(X_test)
tree_measures = [accuracy_score(Y_test, y_pred)]


#KNN Routine

f_measures = pd.DataFrame([],
                  index = [1,2,3,4,5,6,7,8,9,10],
                  columns = ['K', 'Accuracy'])

for x in range(1,11):
    f_measures.loc[x] = get_acc_for_k(X_train, X_test, Y_train, Y_test, x)

#Best K for accuracy
sns.set(style="darkgrid")  
sns.barplot(x='K',y='Accuracy',data=f_measures)
plt.savefig('image3.png')


#Comparing accuracy of Tree vs KNN
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
knn_measures = [accuracy_score(Y_test, y_pred)]

tree_knn_measures = pd.DataFrame([],
        index = [1,2],
        columns = ['Accuracy', 'Classifier'])
tree_knn_measures.loc[1] = tree_measures + ["Tree"]
tree_knn_measures.loc[2] = knn_measures + ["KNN"]

sns.barplot(x='Classifier',y='Accuracy',data=tree_knn_measures)
plt.savefig('image5.png')




