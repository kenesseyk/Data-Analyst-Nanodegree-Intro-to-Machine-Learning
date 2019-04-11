#!/usr/bin/python

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import precision_score, recall_score, accuracy_score

import sys
import pickle
import pandas as pd
import numpy as np
import pprint as pp

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import tester

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [
        'poi',
        'from_messages',
        'from_poi_to_this_person',
        'from_this_person_to_poi',
        'shared_receipt_with_poi',
        'to_messages',
        'bonus',
        'deferred_income',
        'exercised_stock_options',
        'expenses',
        'long_term_incentive',
        'other',
        'restricted_stock',
        'salary',
        'total_payments',
        'total_stock_value']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

identified_outliers = ["TOTAL", "THE TRAVEL AGENCY IN THE PARK"]

for outlier in identified_outliers:
    data_dict.pop(outlier)
 
### Task 3: Create new feature(s)
def add_new_features(data):
    # Convert the data_dict to pandas DataFrame for easier calculations
    df = pd.DataFrame(data_dict)
    df = df.transpose()
    df.replace(to_replace='NaN', value=np.nan, inplace=True)

    
    # Replace missing values with 0 in features
    df[features_list] = df[features_list].fillna(0)

    # Create our new features
    df['fraction_short_term'] = (df['salary'] + df['bonus'] + df['expenses']) / df['total_payments']
    df['fraction_long_term'] = (df['long_term_incentive'] + df['deferred_income']) / df['total_payments']
    df['fraction_salary'] = df['salary']/df['total_payments']
    df['fraction_bonus'] = df['bonus']/df['total_payments']
    df['fraction_long_term_incentive'] = df['salary']/df['long_term_incentive']
    df['fraction_deferred_income'] = df['salary']/df['deferred_income']
    df['fraction_total_stock'] = df['total_stock_value']/df['total_payments']
    df['fraction_to_poi'] = df['from_this_person_to_poi']/df['to_messages']
    df['fraction_from_poi'] = df['from_poi_to_this_person']/df['from_messages']

    # Convert back to the data_dict
    df.replace(np.inf, value=np.nan, inplace=True)
    df.replace(to_replace=np.nan, value='NaN', inplace=True)
    return df.transpose().to_dict()
    
### Store to my_dataset for easy export below.
my_dataset = add_new_features(data_dict)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# SVM Model
from sklearn import svm
svm = Pipeline([('scaler',MinMaxScaler()), ('k_best',SelectKBest()), ('svm',svm.SVC())])
param_grid = ([{'svm__C': [1,50,100,1000],
                'svm__gamma': [0.5, 0.1, 0.01],
                'svm__degree':[1,2],
                'svm__kernel': ['rbf','poly'], 
                'k_best__k':[5,10,15,'all']}])

svm_clf = GridSearchCV(svm, param_grid, scoring='recall').fit(features, labels).best_estimator_

tester.test_classifier(svm_clf, data_dict, features_list)

# Decision Tree Classifier
dt = Pipeline([('scaler',MinMaxScaler()),('kbest',SelectKBest()), ('dt',DecisionTreeClassifier())])
param_grid = ([{'dt__criterion': ['gini','entropy'],
                'dt__splitter':  ['best','random'],
                'dt__min_samples_split': [2,5,10,20,30],
                'kbest__k':[5,10,15,'all']}])
dt_clf = GridSearchCV(dt, param_grid, scoring='recall').fit(features, labels).best_estimator_

tester.test_classifier(dt_clf, data_dict, features_list)

# K-Nearest Neighbors
knn = Pipeline([('scaler', MinMaxScaler()), ('kbest',SelectKBest()), ('knn', KNeighborsClassifier())])
param_grid = ([{'knn__n_neighbors': [2,3,4,5,6], 
                'kbest__k':[5,10,15,'all']}])
knn_clf = GridSearchCV(knn, param_grid, scoring='recall').fit(features, labels).best_estimator_

tester.test_classifier(knn_clf, data_dict, features_list)

# Naive Bayes
nb = Pipeline([('scaler', MinMaxScaler()), ('kbest',SelectKBest()), ('nb', GaussianNB())])
param_grid = ([{'kbest__k':[5,10,15,'all']}])
nb_clf = GridSearchCV(nb, param_grid, scoring='recall').fit(features, labels).best_estimator_

tester.test_classifier(nb_clf, data_dict, features_list)

# Provided to give you a starting point. Try a variety of classifiers.


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
features_list = [
        'poi',
        'bonus',
        'deferred_income',
        'exercised_stock_options',
        'expenses',
        'long_term_incentive',
        'other',
        'restricted_stock',
        'salary',
        'total_payments',
        'total_stock_value',
        'from_messages',
        'from_poi_to_this_person',
        'from_this_person_to_poi',
        'shared_receipt_with_poi',
        'to_messages'
]

data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
feature_list = features_list[1:]


svm = Pipeline([('scaler',StandardScaler()),("kbest", SelectKBest()),('svm',SVC())])
param_grid = ([{'svm__C': [1000],
                'svm__gamma': [0.1],
                'svm__degree':[2],
                'svm__kernel': ['poly'],
                'kbest__k':['all']}])
clf = GridSearchCV(svm, param_grid, scoring='recall').fit(features, labels).best_estimator_

tester.test_classifier(clf, data_dict, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
