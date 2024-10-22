# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 00:33:00 2024

@author: Favour
"""
import pandas as pd
import numpy as np
import timeit
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.metrics import roc_curve, auc
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle
from tensorflow.keras.utils import plot_model

# Loading the dataset
df = pd.read_csv(r'data_with_new_features.csv', delimiter = ',')


def model_training(clf, X_tr, y_tr, X_te, y_te, phase, svc=False):
    '''A function to train a model and evaluate it based on various metrics'''
    #Calculate start time
    start_train = timeit.default_timer()

    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_tr,y_tr)

    #Calculate Stop time for training
    stop_train = timeit.default_timer()
    train_time= stop_train - start_train
    
    
#     #Calculate start time for prediction
#     start = timeit.default_timer()

#     # Predict the model
#     pred_class =clf.predict(X_te)

#     #Calculate Stop time
#     stop = timeit.default_timer()
#     test_time= stop - start
    
    print('Train Time(s):', train_time)

#     #Test time
#     print('Test Time(s):',test_time)
    
    train_pred = clf.predict(X_tr)
    
    train_acc = accuracy_score(y_tr, train_pred)
    
    print('Training Accuracy:  {}%'.format(round(train_acc, 2)*100))
    
    model_evaluation(clf, X_te, y_te, phase,svc)
    
    
def model_evaluation(clf, X_te, y_te, phase, svc=False):
    print('\n Model Evaluation Result For the {} Phase \n'.format(phase))
    #Calculate start time for prediction
    start = timeit.default_timer()
        # Predict the model
    pred_class =clf.predict(X_te)

    #Calculate Stop time
    stop = timeit.default_timer()
    test_time= stop - start
    
#     print('Train Time(s):', train_time)

    #Test time
    print('Test Time(s):',test_time)
    
#     train_pred = clf.predict(X_tr)
    
#     train_acc = accuracy_score(y_tr, train_pred)
    
#     print('Training Accuracy:  {}%\n'.format(round(train_acc, 2)*100))
    
    accuracy = accuracy_score(y_te, pred_class)
    
    report = classification_report(y_te, pred_class)
    
    print('Accuracy:  {}%\n'.format(round(accuracy, 2)*100))
    
    print(report, '\n')
    
    cf_matrix = confusion_matrix(y_te, pred_class)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(
        value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(
        group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show()
    
    if svc:
        y_pred_proba = clf.decision_function(X_te)
    else:
        y_pred_proba = clf.predict_proba(X_te)[::,1]
    # Calculate False Positive Rate, True Positive Rate, and thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    # Calculate the AUC (Area Under the Curve)
    roc_auc = auc(fpr, tpr)
    
    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Dashed diagonal line for random guessing
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
def model_tuning(clf, parameters, train_feat, train_labels, val_feat, val_labels, cv, svc = False):
    grid_search = GridSearchCV(clf, parameters,cv=cv,
                               n_jobs=-1, scoring='accuracy', verbose=1)
    grid_search.fit(train_feat, train_labels)
    model_training(grid_search.best_estimator_, train_feat, train_labels, val_feat, val_labels, phase = 'Tuning', svc = svc)
    return grid_search.best_estimator_

from sklearn.model_selection import cross_val_score

cv = 5
def cross_validation(clf_model, features, labels, cv):
    scores = cross_val_score(
      clf_model, features,
      labels,
      scoring="accuracy", cv=cv)
    acc_scores = scores
    print("Scores:", acc_scores)
    print("Mean:", acc_scores.mean())
    print("StandardDeviation:", acc_scores.std())

# Splitting the dataset to hold the independent and target variables where X represents the independent variable and y the target variable.
X = df.drop('Target', axis=1)
y = df['Target']
y = y.replace({'Dropout': 1, 'Enrolled': 0, 'Graduate': 0})

# Splitting the dataset into train, validation and test dataset 60%-20%-20%
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=105)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=105)

x_eval = pd.concat([x_train, x_val])
y_eval = pd.concat([y_train, y_val])

#scikit_model_pipeline is the name of the scikit learn model pipeline
model_training('scikit_model_pipeline', x_train, y_train, x_val, y_val, phase = 'Training')
param_grid = {
    ''' A dictionary that contains parameters to tune '''
    }

best_model = model_tuning('scikit_model_pipeline', parameters = param_grid, train_feat = x_train,
                              train_labels = y_train, val_feat = x_val, val_labels = y_val, cv = 5)

cross_validation(best_model, x_eval, y_eval, cv)

model_evaluation(best_model, x_test, y_test, phase = 'Test')
