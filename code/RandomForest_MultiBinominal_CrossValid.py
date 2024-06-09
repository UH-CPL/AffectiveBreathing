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
clf = RandomForestClassifier(n_estimators=100, random_state=42)

csvname = 'featuresNew_12_cross.csv'
# # csvname = '..\\DEAP_Dataset\\featuresNew_10_cross_new.csv' #combine arousal and valence
# # csvname = '..\\DEAP_Dataset\\featuresNew_11_cross_new_arousal.csv' #arousal only, 3.3, 6.67
# # csvname = '..\\DEAP_Dataset\\featuresNew_11_cross_new_dominance.csv' #arousal only, 3.3, 6.67
# # csvname = '..\\DEAP_Dataset\\featuresNew_11_cross_new_liking.csv' #arousal only, 3.3, 6.67
# # csvname = '..\\DEAP_Dataset\\featuresNew_11_cross_new_valence.csv' #arousal only, 3.3, 6.67
# csvname = '..\\DEAP_Dataset\\featuresNew_11_cross_new_arousal_7_3.csv' #arousal 5
# # csvname = '..\\DEAP_Dataset\\featuresNew_11_cross_new_arousal_5.csv' #arousal 5
# # csvname = '..\\DEAP_Dataset\\featuresNew_13_cross_clustermarker.csv' #arousal 5
# # csvname = '..\\DEAP_Dataset\\featuresNew_14_cross_clustermarker.csv' #arousal 5* 0.31 - valence * 0.1 - 0 .2
# # csvname = '..\\DEAP_Dataset\\featuresNew_16_cross_labels.csv'
# # csvname = '..\\DEAP_Dataset\\featuresNew_16_cross_labels_by_video1.csv'
# csvname = '..\\case_Dataset\\caseNew_1_cross_NO_3.csv' #arousal 5
# csvname = '..\\case_Dataset\\caseNew_2_cross.csv' #arousal 5
# # csvname = '..\\case_Dataset\\caseNew_3_cross.csv'
# # csvname = '..\\case_Dataset\\caseNew_4_cross.csv'
# csvname = '..\\case_Dataset\\new\\caseNew_5_cross_1221.csv'
# csvname = '..\\..\\data_NoPMRB\\featuresNew_9_cross.csv'
csvname = 'featuresNew_12_cross.csv'
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
# print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
# print('X length', len(X))

# from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def over_sample(data_final):
    #*******over sample the NO data records, so the yes and no samples *******
    # are having balanced numbers
    X = data_final.loc[:, data_final.columns != targetCol]
    y = data_final.loc[:, data_final.columns == targetCol]
    #PYTHON 3.8
    #pip install -U imbalanced-learn
    #conda install -c conda-forge imbalanced-learn

    os = SMOTE(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    # print('X_train size:', len(X_train), 'X_test size:', len(X_test))
    columns = X_train.columns
    os_data_X,os_data_y=os.fit_resample(X_train, y_train)
    os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
    os_data_y= pd.DataFrame(data=os_data_y,columns=[targetCol])
    # we can Check the numbers of our data
    print("length of oversampled data is ",len(os_data_X))
    print("Number of no subscription in oversampled data",len(os_data_y[os_data_y[targetCol]==0]))
    print("Number of subscription",len(os_data_y[os_data_y[targetCol]==1]))
    print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y[targetCol]==0])/len(os_data_X))
    print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y[targetCol]==1])/len(os_data_X))
    
    return os_data_X, os_data_y

# oversample
os_data_X, os_data_y = over_sample(data_final)
X=os_data_X
y=os_data_y

# print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
# print('oversampled X length', len(X))

cv = ShuffleSplit(n_splits=3, test_size=0.33, random_state=0)
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

#prepare multi-binominal
y = y.values.ravel()
myY1 = deepcopy(y)
myY2 = deepcopy(y)
myY3 = deepcopy(y)
myY=[myY1,myY2,myY3]

#0 is PR, 1 is ST, 2 is LT
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
print('LT 1 count is:', count_1)
print('LT 0 count is:', count_0)

from sklearn.model_selection import RandomizedSearchCV, train_test_split
#from scipy.stats import randint
# Split the data into training and test sets
# predict PR (0)

sessions = ['PR', 'ST', 'LT']

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def find_best_f1_threshold(y_test, y_score):
    max_f1 = 0
    best_threshold = 0
    precision, recall, threshold = precision_recall_curve(y_test, y_score, pos_label=1)
    for fpr_item, tpr_item, threshold_item in zip(precision, recall, threshold):
        if tpr_item+fpr_item == 0:
            continue
        f1 = 2*(tpr_item*fpr_item)/(tpr_item+fpr_item)
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        # print('precision', fpr_item, 'recall', tpr_item,'f1',f1)
        if f1 > max_f1:
            max_f1 = f1
            best_threshold = threshold_item
        # print('f1=',f1, "{:.2f}".format(fpr_item), "{:.2f}".format(tpr_item), "{:.2f}".format(threshold_item))
        
    fpr, tpr, threshold = roc_curve(y_test, y_score, pos_label=1)
    best_tpr = best_fpr = 0
    threhold_min_diff = 100.0
    
    for fpr_item, tpr_item, threshold_item in zip(fpr, tpr, threshold):
        threhold_diff = abs(threshold_item - best_threshold)
        if threhold_diff < threhold_min_diff:
            threhold_min_diff = threhold_diff
            best_tpr = tpr_item
            best_fpr = fpr_item
            # print('fpr:', "{:.4f}".format(fpr_item),'tpr', "{:.4f}".format(tpr_item), "{:.2f}".format(threshold_item))
    
    y_test_last_col = y_test[:].tolist()
    print(len(y_test_last_col),y_test_last_col.count(0),y_test_last_col.count(1))
    P = y_test_last_col.count(1) #P=TP+FN
    N = y_test_last_col.count(0) #N=TN+FP
    
    TP = best_tpr*P
    FP = best_fpr*N
    TN = N - FP
    FN = P - TP
    
    return TP,FP,TN,FN, best_threshold

def plot_feature_importance(forest):
    figureName = 'RandomForest_OfficeTask_feat_importance'
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)
    forest_importances_err = pd.Series(std, index=feature_names)
    x = forest_importances.nlargest(8)
    print(x)
    # print(forest_importances_err)
    
    # fig, ax = plt.subplots()
    # forest_importances.plot.bar(yerr=std, ax=ax)
    # ax.set_title("Feature importances using MDI")
    # ax.set_ylabel("Mean decrease in impurity")
    # fig.tight_layout()
    # plt.grid(False)    
    
    # print(figureName+' saved!')
    # plt.savefig(figureName, format="png",dpi=600)
    # plt.show()

def rand_cross(y, sessionString, fig, ax, treatment_index):
    colorMean = ["r","g","b"]
    treatments = ['PR','ST','LT']
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
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
        print(treatments[treatment_index], '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        # print(treatments[treatment_index],'x_train size:', len(X_train), 'x_test size:', len(my_X_test))
        y_pred = rf.predict(my_X_test) 
        accuracy = accuracy_score(my_Y_test, y_pred)
        precision = precision_score(my_Y_test, y_pred)
        recall = recall_score(my_Y_test, y_pred)
        f1_score = (2 * precision * recall) / (precision + recall)
        # print("Accuracy:", accuracy, 'Precision', precision, 'Recall',recall, 'f1 score', f1_score)
        
        y_score = rf.predict_proba(my_X_test)[:,1]
        # print('length of y_score', len(y_score))
        # print('y_score',y_score)
        TP,FP,TN,FN, best_threshold = find_best_f1_threshold(my_Y_test, y_score)
        print('TP',TP,'FP',FP,'TN',TN,'FN',FN,'best threshold', best_threshold)
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        f1 = 2*precision*recall/(precision+recall)
        print(treatments[treatment_index],'recall',recall,'precision',precision,'accuracy',accuracy,'f1',f1)
        
        
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color=colorMean[treatment_index],
        label=r"%s Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (treatments[treatment_index], mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    if treatment_index == 0:
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color=colorMean[treatment_index],
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
    else:
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color=colorMean[treatment_index],
            alpha=0.2,
        )
        
    
    return accuracy, mean_auc


def rand_predict(y, sessionString):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, randon_state=0)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_score = (2 * precision * recall) / (precision + recall)
    print(sessionString, "Accuracy:", accuracy, 'Precision', precision, 'Recall',recall, 'f1 score', f1_score)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print('TP:', tp, 'TN:',tn, 'FP:',fp,'FN:',fn)
    
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)
    ax = plt.gca()
    # rfc_disp = RocCurveDisplay.from_estimator(rfc, X_test, y_test, ax=ax, alpha=0.8)
    # svc_disp.plot(ax=ax, alpha=0.8)
    # plt.show()
    
    #print out feature importance
   #  (pd.Series(rfc.feature_importances_, index=X.columns)
   # .nlargest(6)
   # .plot(kind='barh'))  

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
        # rand_predict(myY[i],sessions[i])
    
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
            
    plt.show()
    print(accuracy_mean)
    break
    if auc_mean > 0.635:
        break


