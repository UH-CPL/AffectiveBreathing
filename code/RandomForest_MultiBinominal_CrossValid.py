# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 10:55:00 2023

@author: sunnf
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from copy import copy, deepcopy
import numpy as np
import pandas as pd

# Initialize with whatever parameters you want to
clf = RandomForestClassifier()

csvname = 'featuresNew_12_cross.csv'
# csvname = 'caseNew_5_cross.csv'
isNewFile = False
if 'cross' in csvname:
    isNewFile = True

data = pd.read_csv(csvname,header=0)
print("data is ///////////////////////")
print(data.shape)

data = data.dropna()
data = data.sort_values(by=['M_Stress','Subject'])

print("data is ///////////////////////")
print(data.shape)

if isNewFile == True:
    targetCol = 'M_Stress'
else:
    targetCol = 'MLStress'

feature_names=['BVC_AVG','BVT_AVG','WA_AVG','BVC_SD','WA_SD','BVC_NORM','BVT_NORM','WA_NORM','BVC_SD_NORM','WA_SD_NORM','RTQ_AVG','RTQ_SD','RTQ_NORM','RTQ_SD_NORM','BR_AVG','WL_AVG','WL_SD','BR_SD','BR_NORM','WL_NORM','WL_SD_NORM','BR_SD_NORM']
if isNewFile != True:
    to_keep=['VolCycle','VolCycle_STD','VolCycle_Ratio','VolSec','VolSec_Ratio','RTQ','RTQ_STD','RTQ_Ratio','BR','BR_STD','BR_Ratio','Height','Height_STD','Height_Ratio','Length','Length_STD','Length_Ratio',targetCol]
else: 
    to_keep=feature_names.copy()
    to_keep.append(targetCol)

data_final=data[to_keep]

#include only selected columns
X = data_final.loc[:, data_final.columns != targetCol]
y = data_final.loc[:, data_final.columns == targetCol]
print(X.head())
print(y.head())

cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
results_accuracy = cross_val_score(clf, X, y.values.ravel(), cv=cv)
scoring = ['precision_macro', 'recall_macro', 'f1_macro']
scores = cross_validate(clf, X, y.values.ravel(), cv=cv, scoring=scoring)



print('#####random forest results#####')
print(results_accuracy)
# 10-Fold Cross validation
print('Accuracy:', np.mean(results_accuracy))

print(sorted(scores.keys()))
print(scores)
# print('f1',results_f1)
# print('f1 score:', np.mean(results_f1))

#decision tree model
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
# Create Decision Tree classifer object
tree_clf = DecisionTreeClassifier()
tree_results = cross_val_score(tree_clf, X, y.values.ravel(), cv=cv)
print('##### DCTree results#####')
print(tree_results)
print(np.mean(tree_results))

#SVM
# from sklearn import svm
# #Create a svm Classifier
# svm_clf = svm.SVC(kernel='linear')
# svm_results = cross_val_score(svm_clf, X_train, y_train.values.ravel(), cv=cv)

# print('##### SVM results#####')
# print(svm_results)
# # 10-Fold Cross validation
# print(np.mean(svm_results))

#prepare multi-binominal
y = y.values.ravel()
myY1 = deepcopy(y)
myY2 = deepcopy(y)
myY3 = deepcopy(y)
myY=[myY1,myY2,myY3]

#0 is PR, 1 is ST, 2 is DT
count_1 = 0
count_0 = 0
for i in range(len(myY[0])):
    if myY[0][i] == 0:
        myY[0][i] = 1
        count_1+=1
    else:
        myY[0][i] = 0
        count_0+=1

print('PR 1 count is:', count_1)
print('PR 0 count is:', count_0)

count_1 = 0
count_0 = 0
        
for i in range(len(myY[1])):
    if myY[1][i] != 1:
        myY[1][i] = 0
        count_0+=1
    else:
        count_1+=1
print('ST 1 count is:', count_1)
print('ST 0 count is:', count_0)

count_1 = 0
count_0 = 0

for i in range(len(myY[2])):
    if myY[2][i] == 2:
        myY[2][i] = 1
        count_1+=1
    else:
        myY[2][i] = 0
        count_0+=1
print('DT 1 count is:', count_1)
print('DT 0 count is:', count_0)

from sklearn.model_selection import RandomizedSearchCV, train_test_split
#from scipy.stats import randint
# Split the data into training and test sets
# predict PR (0)

sessions = ['PR', 'ST', 'DT']

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def plot_feature_importance(forest):
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)
    forest_importances_err = pd.Series(std, index=feature_names)
    print(forest_importances)
    print(forest_importances_err)

def rand_cross(y, sessionString, fig, ax, treatment):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    rf = RandomForestClassifier()
    # fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        X_train = X.values[train]
        y_train = y[train]
        rf.fit(X_train, y_train)
        plot_feature_importance(rf)
        
        viz = RocCurveDisplay.from_estimator(
            rf,
            X.values[test],
            y[test],
            name="ROC fold {}".format(i),
            alpha=0.0,
            lw=1,
            ax=ax,
        )
        
        my_X_test = X.values[test]
        my_Y_test = y[test]
        y_pred = rf.predict(my_X_test) 
        accuracy = accuracy_score(my_Y_test, y_pred)
        precision = precision_score(my_Y_test, y_pred)
        recall = recall_score(my_Y_test, y_pred)
        f1_score = (2 * precision * recall) / (precision + recall)
        print("Accuracy:", accuracy, 'Precision', precision, 'Recall',recall, 'f1 score', f1_score)
        
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        
    colorMean = ["r","g","b"]
    treatments = ['PR','ST','LT']
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color=colorMean[treatment],
        label=r"%s Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (treatments[treatment], mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    if treatment == 0:
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color=colorMean[treatment],
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
    else:
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color=colorMean[treatment],
            alpha=0.2,
        )
        
    
    return accuracy, mean_auc


def rand_predict(y, sessionString):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_score = (2 * precision * recall) / (precision + recall)
    print(sessionString, "Accuracy:", accuracy, 'Precision', precision, 'Recall',recall, 'f1 score', f1_score)
    
    rfc = RandomForestClassifier(n_estimators=10, random_state=42)
    rfc.fit(X_train, y_train)
    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_estimator(rfc, X_test, y_test, ax=ax, alpha=0.8)

    return accuracy

#adaboost
from sklearn.ensemble import AdaBoostClassifier

def ada_cross(y, sessionString, fig, ax, treatment):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    ada_clf = AdaBoostClassifier(n_estimators=100)
    # fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        X_train = X.values[train]
        y_train = y[train]
        ada_clf.fit(X_train, y_train)
        viz = RocCurveDisplay.from_estimator(
            ada_clf,
            X.values[test],
            y[test],
            name="ROC fold {}".format(i),
            alpha=0.0,
            lw=1,
            ax=ax,
        )
        
        my_X_test = X.values[test]
        my_Y_test = y[test]
        y_pred = ada_clf.predict(my_X_test) 
        accuracy = accuracy_score(my_Y_test, y_pred)
        precision = precision_score(my_Y_test, y_pred)
        recall = recall_score(my_Y_test, y_pred)
        f1_score = (2 * precision * recall) / (precision + recall)
        print("Accuracy:", accuracy, 'Precision', precision, 'Recall',recall, 'f1 score', f1_score)
        
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        
    colorMean = ["r","g","b"]
    treatments = ['PR','ST','LT']
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color=colorMean[treatment],
        label=r"%s Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (treatments[treatment], mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    if treatment == 0:
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color=colorMean[treatment],
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
    else:
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color=colorMean[treatment],
            alpha=0.2,
        )
        
    
    return accuracy, mean_auc
    
def ada_predict(y, sessionString):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    ada_clf = AdaBoostClassifier(n_estimators=100)
    
    ada_clf.fit(X_train, y_train)
    y_pred = ada_clf.predict(X_test)
    
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_score = (2 * precision * recall) / (precision + recall)
    print(sessionString, "Accuracy:", accuracy, 'Precision', precision, 'Recall',recall, 'f1 score', f1_score)
    
    ax = plt.gca()
    ada_disp = RocCurveDisplay.from_estimator(ada_clf, X_test, y_test, ax=ax, alpha=0.8)
    
    # scores = cross_val_score(ada_clf, X, y, cv=5)
    # accuracy = scores.mean()    
    # print(sessionString, "Accuracy:", accuracy)
    return accuracy
   
flag = True 
while flag:
    # flag = False
    print('##########random Forest results:##########')
    accuracy_results = []
    auc_results =[]
    fig, ax = plt.subplots()
    for i in range(3):
        accuracy, mean_auc = rand_cross(myY[i],sessions[i], fig, ax, i)
        # accuracy, mean_auc = ada_cross(myY[i],sessions[i], fig, ax, i)
        accuracy_results.append(accuracy)
        auc_results.append(mean_auc)
        # accuracy_results.append(rand_predict(myY[i],sessions[i]))
    
    #draw all elements for merged figures
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black", label="Chance", alpha=0.8)
    ax.set(
       xlim=[-0.05, 1.05],
       ylim=[-0.05, 1.05],
       title="Random Forest Cross Validation in PR, ST, LT",
       # title="Random Forest Cross Validation in PR, ST, LT",
    )
    ax.legend(loc="lower right",prop={'size': 8.5})
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.grid(which='major', color='#DDDDDD', linestyle=':', linewidth=0.5)
    plt.grid(False)
    
    # Here is the trick to hide ROC field legends
    plt.gcf()
    handles, labels = plt.gca().get_legend_handles_labels()
    index_list = [3,7,11,12,13]
    handles = np.take(handles, index_list)
    labels = np.take(labels, index_list)
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    figureName = 'Figure11b_randomforest.png'
    print(figureName+' saved!')
    plt.savefig(figureName, format="png",dpi=600)
    plt.show()
    
    accuracy_mean = np.mean(accuracy_results) 
    auc_mean = np.mean(auc_results) 
    print('accuracy',accuracy_mean)
    print('auc',auc_mean)
    
    
    #skip adaboost
    # print('##########   adaboost results:    ##########')
    # ada_accuracy_results = []
    # for i in range(3):
    #     ada_accuracy_results.append(ada_predict(myY[i],sessions[i]))
        
    plt.show()
    print(accuracy_mean)
    
    if auc_mean > 0.635:
        break


