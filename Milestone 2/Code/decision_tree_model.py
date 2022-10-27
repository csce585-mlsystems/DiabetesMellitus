import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

current_path_str = os.getcwd()
current_path_list = current_path_str.split("/")
dataset_path_list = current_path_list[:-1]
dataset_path_list.append("Dataset")
dataset_path_str = "/".join(dataset_path_list)
path = dataset_path_str + "/diabetes.csv"

# load the dataset to pandas dataframe
data_dec_tree = pd.read_csv(path)

#show attributes and their types so we know what we're working with
data_dec_tree.dtypes

#show first 5 lines to peek at data
data_dec_tree.head()


Xs = data_dec_tree.values[:,1:5]
Ys = data_dec_tree.values[:,0]

#spliting the data into
X_training, X_testing, Y_training, Y_testing = train_test_split(Xs,Ys, test_size = 0.3, random_state = 75)

#Training using entropy method

#Traning using giniIndex:
    #first, pick data point
    #classify it by the outcome
clf_gini = DecisionTreeClassifier(criterion="gini",random_state=100, max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_training, Y_training)
predict_gini = clf_gini.predict(X_testing)
print("Accuracy : ", accuracy_score(Y_testing,Y_training)*100)
#print("Classification report : ", classification_report(Y_testing,Y_training))
