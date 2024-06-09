# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 12:59:15 2022

@author: sunnf
"""
#https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
import pandas as pd
import numpy as np
from copy import copy, deepcopy
from scipy import stats
from sklearn import preprocessing
import matplotlib.pyplot as plt 

hasGridline = False

plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

#data = pd.read_csv('..\\..\\data_NoPM\\RegressionDataSet_NoPM.csv',header=0)
#data = pd.read_csv('..\\..\\data_NoPMRB\\RegressionDataSetNew_Cleaned_NoPMRB_AllFeature_Treatment.csv',header=0)
# data = pd.read_csv('..\\..\\data_NoPMRB\\features_threshold5_NoDB.csv',header=0)
# data = pd.read_csv('..\\..\\data_NoPMRB\\features_threshold5_NoDB_3.csv',header=0)
#                                        #RegressionDataSetNew_Cleaned_NoPMRB_AllFeature_Treatment
# data = pd.read_csv('..\\..\\data_NoPMRB\\features_threshold5_NoDB_3.csv',header=0)
# data = pd.read_csv('..\\..\\data_NoPMRB\\featuresNew_raw_4.csv',header=0)
# data = pd.read_csv('..\\..\\data_NoPMRB\\featuresNew_raw_5.csv',header=0)
csvname = '..\\..\\data_NoPMRB\\featuresNew_raw_5.csv'
# data = pd.read_csv('..\\..\\data_NoPMRB\\featuresNew_raw_7.csv',header=0)
csvname = '..\\..\\data_NoPMRB\\featuresNew_8_cross.csv'
csvname = '..\\..\\data_NoPMRB\\featuresNew_9_cross_Clean.csv'
csvname = '..\\..\\data_NoPMRB\\featuresNew_9_cross.csv'
# csvname = '..\\..\\data_NoPMRB\\featuresNew_10_cross.csv'
# csvname = '..\\DEAP_Dataset\\featuresNew_11_cross_new_arousal.csv'
csvname = '..\\..\\data_NoPMRB\\featuresNew_12_cross_CleanedNoBaseline.csv'
csvname = 'featuresNew_12_cross.csv'
isNewFile = False
if 'cross' in csvname:
    isNewFile = True
    
data = pd.read_csv(csvname,header=0)
#log transform wav_avg
print("original data shape is ///////////////////////")
print(data.shape)

if 'WL_AVG' in data.columns:
    data['WL_AVG_LOG'] = np.log2(data['WL_AVG'])



data = data.dropna()
#convert data frame to derive standized Beta coefficients which is not working
#https://stackoverflow.com/questions/50842397/how-to-get-standardised-beta-coefficients-for-multiple-linear-regression-using
#data = data.select_dtypes(include=[np.number]).dropna().apply(stats.zscore)
print("data is ///////////////////////")
print(data.shape)
print(list(data.columns))
print(data.head(5))

if isNewFile == True:
    targetCol = 'M_Stress'
else:
    targetCol = 'MLStress'
    
#print columns 'y' represents 1 and 0 rows
print(data[targetCol].value_counts())
sns.countplot(x=targetCol,data=data,palette='hls')
plt.show()
plt.savefig('count_plot')

cat_vars=['Subject','Treatment']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
#selected 8 features
if isNewFile != True:
    #to_keep=['Height_STD','VolSec_Ratio','RTQ','RTQ_STD','RTQ_Ratio','BR','BR_STD', 'BR_Ratio','Mean',targetCol]
    to_keep=['VolCycle','VolCycle_STD','VolCycle_Ratio','VolSec','VolSec_Ratio','RTQ','RTQ_STD','RTQ_Ratio','BR','BR_STD','BR_Ratio','Height','Height_STD','Height_Ratio','Length','Length_STD','Length_Ratio',targetCol]
    to_keep_no_target=['VolCycle','VolCycle_STD','VolCycle_Ratio','VolSec','VolSec_Ratio','RTQ','RTQ_STD','RTQ_Ratio','BR','BR_STD','BR_Ratio','Height','Height_STD','Height_Ratio','Length','Length_STD','Length_Ratio']
else: 
    to_keep=['BVC_AVG','BVT_AVG','WA_AVG','BVC_SD','WA_SD','BVC_NORM','BVT_NORM','WA_NORM','BVC_SD_NORM','WA_SD_NORM','RTQ_AVG','RTQ_SD','RTQ_NORM','RTQ_SD_NORM','BR_AVG','WL_AVG','WL_AVG_LOG','WL_SD','BR_SD','BR_NORM','WL_NORM','WL_SD_NORM','BR_SD_NORM',targetCol]
    to_keep_no_target=['BVC_AVG','BVT_AVG','WA_AVG','BVC_SD','WA_SD','BVC_NORM','BVT_NORM','WA_NORM','BVC_SD_NORM','WA_SD_NORM','RTQ_AVG','RTQ_SD','RTQ_NORM','RTQ_SD_NORM','BR_AVG','WL_AVG','WL_AVG_LOG','WL_SD','BR_SD','BR_NORM','WL_NORM','WL_SD_NORM','BR_SD_NORM']
    
data_final=data[to_keep]
# data_final.columns.values
print("data_final is ///////////////////////")
print(list(data_final.columns))


# from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
    
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
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('X_train size:', len(X_train), 'X_test size:', len(X_test))
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

os_data_X, os_data_y = over_sample(data_final)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize

######################################
#generate intial regression results with all features


if isNewFile == True:
    print('***************Processing new file******************')
    cols = ['WA_AVG','BVT_AVG','BVC_AVG','BVC_SD','WA_SD','BVC_NORM','BVT_NORM','WA_NORM','RTQ_AVG','RTQ_SD','RTQ_NORM','BR_AVG','WL_AVG','WL_SD','BR_SD','BR_NORM','WL_NORM']
    cols = [         'BVT_AVG',          'BVC_SD','WA_SD',           'BVT_NORM','WA_NORM','RTQ_AVG','RTQ_SD','RTQ_NORM',         'WL_AVG','WL_SD','BR_SD','BR_NORM','WL_NORM']
    cols = [                             'BVC_SD','WA_SD',           'BVT_NORM','WA_NORM','RTQ_AVG','RTQ_SD','RTQ_NORM',         'WL_AVG','WL_SD','BR_SD','BR_NORM']
    cols = [                             'BVC_SD','WA_SD',           'BVT_NORM','WA_NORM','RTQ_AVG','RTQ_SD','RTQ_NORM',         'WL_AVG','WL_SD','BR_SD','BR_NORM']
    cols = [                             'BVC_SD','WA_SD',           'BVT_NORM','WA_NORM','RTQ_AVG','RTQ_SD','RTQ_NORM',         'WL_AVG','WL_SD','BR_SD']
    cols = [                             'BVC_SD','WA_SD',           'BVT_NORM','WA_NORM','RTQ_AVG','RTQ_SD','RTQ_NORM',         'WL_AVG','WL_SD']
    cols = [                             'BVC_SD','WA_SD',           'BVT_NORM',          'RTQ_AVG','RTQ_SD','RTQ_NORM',         'WL_AVG','WL_SD']
    cols = [                             'BVC_SD','WA_SD',           'BVT_NORM',          'RTQ_AVG','RTQ_SD',                   'WL_AVG','WL_SD']
    #cols = ['BVT_NORM','RTQ_AVG','RTQ_SD','BR_SD','WL_NORM']#Best ST
    #cols = ['BVT_AVG','RTQ_AVG', 'WA_SD', 'WL_AVG','WL_SD']
    #cols=['BVT_NORM','RTQ_AVG','WL_AVG','WL_SD','WL_NORM']
    cols = ['WA_AVG','BVT_AVG','BVC_AVG','BVC_NORM','BVT_NORM','WA_NORM','RTQ_AVG','RTQ_NORM','WL_AVG','WL_SD','BR_NORM']
    cols = ['WA_AVG','BVC_AVG','BVT_NORM','RTQ_AVG','WL_AVG','WL_SD']
    #8
    cols = ['WA_AVG','BVC_AVG','BVT_NORM','RTQ_AVG','WL_AVG','WL_SD']
    #9
    cols = ['WA_AVG','BVT_AVG','BVT_NORM','WA_NORM','RTQ_AVG','RTQ_NORM','WL_AVG','BR_NORM']
    cols = ['BVT_NORM','WA_SD','RTQ_AVG','RTQ_NORM','WL_AVG','WL_SD']
    # cols = ['RTQ_AVG','WL_AVG','WL_SD','WA_SD']
    #data.to_csv(csvname+'_log.csv')
    #10
    # cols = ['WA_AVG','BVT_AVG','BVC_NORM','RTQ_AVG','RTQ_NORM','WL_AVG','BR_NORM','WL_SD','WA_SD']
    # cols = ['WA_AVG','BVT_AVG','BVC_NORM','RTQ_NORM','WL_AVG','WL_SD','WA_SD']
    # cols = ['BVT_AVG','RTQ_NORM','WL_AVG','WA_SD']

else:
    print('$$$$$$$$$$$$$$$$$$Processing OLD file$$$$$$$$$$$$$$$$')
    #keep all
    cols=['VolCycle','VolCycle_STD','VolCycle_Ratio','VolSec','VolSec_Ratio','RTQ','RTQ_STD','RTQ_Ratio','BR','BR_Ratio','Height','Height_STD','Height_Ratio','Length','Length_STD','Length_Ratio']
    #Traditional BR
    # cols=['VolCycle','VolCycle_STD','VolCycle_Ratio','VolSec','VolSec_Ratio','RTQ','RTQ_STD','RTQ_Ratio','BR','BR_Ratio','Height','Height_STD',               'Length','Length_STD','Length_Ratio']
    # cols=['VolCycle','VolCycle_STD','VolCycle_Ratio','VolSec','VolSec_Ratio','RTQ','RTQ_STD','RTQ_Ratio',     'BR_Ratio','Height','Height_STD',               'Length','Length_STD','Length_Ratio']
    # cols=[           'VolCycle_STD','VolCycle_Ratio','VolSec','VolSec_Ratio','RTQ','RTQ_STD','RTQ_Ratio',     'BR_Ratio','Height','Height_STD',               'Length','Length_STD']
    # cols=[           'VolCycle_STD','VolCycle_Ratio','VolSec',               'RTQ','RTQ_STD',                            'Height','Height_STD',               'Length','Length_STD']
    # cols=[           'VolCycle_STD',                                         'RTQ','RTQ_STD',                                     'Height_STD',               'Length','Length_STD']
    # cols=['VolCycle_STD','RTQ','RTQ_STD','Height_STD','Length','Length_STD']
    
    
    #Instant BR
    # cols=['VolCycle_STD','VolCycle_Ratio','VolSec','VolSec_Ratio','RTQ','RTQ_STD','BR','BR_Ratio','Height','Height_STD','Height_Ratio','Length','Length_STD','Length_Ratio']
    # cols=['VolCycle_STD','VolCycle_Ratio','VolSec','VolSec_Ratio','RTQ','RTQ_STD','BR',           'Height','Height_STD','Height_Ratio','Length','Length_STD','Length_Ratio']
    # cols=['VolCycle_STD','VolCycle_Ratio','VolSec','VolSec_Ratio','RTQ','RTQ_STD','BR',           'Height','Height_STD',               'Length','Length_STD','Length_Ratio']
    # cols=['VolCycle_STD','VolCycle_Ratio','VolSec','VolSec_Ratio','RTQ',          'BR',           'Height','Height_STD',               'Length','Length_STD','Length_Ratio']
    # cols=[               'VolCycle_Ratio','VolSec','VolSec_Ratio','RTQ',          'BR',           'Height','Height_STD',               'Length','Length_STD','Length_Ratio']
    # cols=[                                'VolSec','VolSec_Ratio','RTQ',                                   'Height_STD',               'Length','Length_STD','Length_Ratio']
    
    #Wednesday model
    cols=['VolCycle_STD','VolSec_Ratio','RTQ','RTQ_STD','BR','BR_STD','Height_STD','Length','Length_STD', 'Length_Ratio']
    #No SD model
    cols=['VolCycle','VolSec','VolCycle_Ratio','VolSec_Ratio','RTQ','RTQ_Ratio','BR','BR_Ratio','Height','Height_Ratio','Length','Length_Ratio']
    cols=['VolCycle','VolCycle_Ratio','VolSec_Ratio','RTQ','RTQ_Ratio','BR','BR_Ratio','Height','Height_Ratio','Length','Length_Ratio']
    cols=['VolCycle','VolCycle_Ratio','VolSec_Ratio','RTQ','BR','BR_Ratio','Height','Height_Ratio','Length','Length_Ratio']
    cols=['VolCycle','VolCycle_Ratio','VolSec_Ratio','RTQ','BR','BR_Ratio','Height','Height_Ratio','Length','Length_STD','Length_Ratio']
    cols=['VolCycle','VolCycle_Ratio','VolSec_Ratio','RTQ','BR','BR_Ratio','Height','Length','Length_STD','Length_Ratio']
    cols=['VolCycle','VolCycle_Ratio','VolSec_Ratio','RTQ','BR','Height','Length','Length_STD','Length_Ratio']
    cols=['VolCycle_Ratio','VolSec_Ratio','RTQ','BR','Length','Length_STD','Length_Ratio']
    cols=['VolSec_Ratio','RTQ','Length','Length_STD','Length_Ratio']
    
    cols_PR = ['VolSec_Ratio','RTQ','RTQ_STD','BR','BR_STD','Height','Height_STD','Height_Ratio','Length_STD','Length_Ratio']#best PR, 0.93, 0.94
    cols_ST = ['VolSec_Ratio','RTQ','RTQ_STD',     'BR_STD',                                                  'Length_Ratio']#best ST, 0.84, 0.80
    cols_DT = ['VolCycle_STD','Height_STD','Length','Length_STD']#best DT, 0.82, 0.80


# oversample
X=os_data_X[cols]
y=os_data_y[targetCol]

# X = data_final[cols]
# y = data_final[targetCol]


print('x length:', len(X))
myY1 = deepcopy(y)
myY2 = deepcopy(y)
myY3 = deepcopy(y)
myY=[myY1,myY2,myY3]


print('*******************************************')
print(len(myY[0]))
print(myY[0].iat[0])

#0 is PR, 1 is ST, 2 is DT
for i in range(len(myY[0])):
    if myY[0].iat[i] == 0:
        myY[0].iat[i] = 1
    else:
        myY[0].iat[i] = 0
        
for i in range(len(myY[1])):
    if myY[1].iloc[i] != 1:
        myY[1].iloc[i] = 0

for i in range(len(myY[2])):
    if myY[2].iloc[i] == 2:
        myY[2].iloc[i] = 1
    else:
        myY[2].iloc[i] = 0

#create a function to derive standard coefficients Beta
def printStandardizedCoef(results, model):
    std = model.exog.std(0)
    std[0] = 1
    tt = results.t_test(np.diag(std))
    print(tt.summary())
    tt.summary_frame()
#print out coefficient and P-values
import statsmodels.api as sm
logit_model=sm.Logit(myY[0],X)
result=logit_model.fit()
printStandardizedCoef(result,logit_model)
#print(result.summary2())
logit_model=sm.Logit(myY[1],X)
result=logit_model.fit()
printStandardizedCoef(result,logit_model)
#print(result.summary2())
logit_model=sm.Logit(myY[2],X)
result=logit_model.fit()
printStandardizedCoef(result,logit_model)
#print(result.summary2())

y = label_binarize(y, classes=[0,1,2])

n_classes = 3

#Logistic Regression Model Fitting
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print('X_train size:', len(X_train), 'X_test size:', len(X_test))

model = LogisticRegression(max_iter=500)
logreg = OneVsRestClassifier(model)
result = logreg.fit(X_train, y_train)
print('?????????????????????????????????????????????')
print('X len:',len(X))
print(result.coef_)
print(result.intercept_)

y_score = logreg.fit(X_train, y_train).decision_function(X_test)
y_pred = logreg.predict(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
from sklearn.metrics import classification_report

print('123451234512345123451234512345123451234512345')

coefficients = logreg.coef_[0]
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(coefficients)})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print('##################logreg feature_importance #####################')
print(feature_importance)

print('##################logreg all data results #####################')
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    #print out model report
    print(classification_report(y_test[:, i], y_pred[:, i]))

# y_pred = logreg.predict(X_test)
# print(classification_report(y_test, y_pred))


print('123451234512345123451234512345123451234512345')

# Plot of a ROC curve for a specific class
plt.figure()

#force the order to PR-0, DT-2, ST-1
xList = [0,2,1]
for i in xList:
    mylabel=''
    if i==0:
        mylabel = '$AR$-PR ROC curve (area = %0.2f)' % roc_auc[i]
    elif i==1:
        mylabel = '$AR$-ST ROC curve (area = %0.2f)' % roc_auc[i]
    elif i==2:
        mylabel = '$AR$-LT ROC curve (area = %0.2f)' % roc_auc[i]
        
    plt.plot(fpr[i], tpr[i], label=mylabel)

# for i in range(n_classes):
#     mylabel=''
#     if i==0:
#         mylabel = 'PR ROC curve (area = %0.2f)' % roc_auc[i]
#     elif i==1:
#         mylabel = 'ST ROC curve (area = %0.2f)' % roc_auc[i]
#     elif i==2:
#         mylabel = 'DT ROC curve (area = %0.2f)' % roc_auc[i]
        
#     plt.plot(fpr[i], tpr[i], label=mylabel)

plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('ROC Curves of Multinomial Results')
plt.legend(loc="lower right")

if hasGridline:
    plt.grid(which='major', color='#DDDDDD', linestyle=':', linewidth=0.5)
else:
    plt.grid(False)

plt.savefig('Figure9.png', format="png",dpi=300)
#plt.savefig('Multinominial_ROC_NoVolume', dpi=300)
plt.show()

# #############################################################################
# Classification and ROC analysis
from sklearn.metrics import RocCurveDisplay,precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

def find_best_fit_2_threshold(y_test, y_score):
    max_f1 = 0
    best_threshold = 0
    precision, recall, threshold = precision_recall_curve(y_test, y_score, pos_label=1)
    org_precision = 0.81
    org_recall = 0.78
    sqr_error_min=10000
    for fpr_item, tpr_item, threshold_item in zip(precision, recall, threshold):
        f1 = 2*(tpr_item*fpr_item)/(tpr_item+fpr_item)
        sqr_error = (fpr_item - org_precision)*(fpr_item - org_precision) + (tpr_item-org_recall)*(tpr_item-org_recall)
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        # print('precision', fpr_item, 'recall', tpr_item,'f1',f1)
        if sqr_error_min > sqr_error:
            sqr_error_min = sqr_error
            best_threshold = threshold_item
            # print('f1=',f1,'precision', "{:.2f}".format(fpr_item),'recall', "{:.2f}".format(tpr_item), "{:.2f}".format(threshold_item),'sqr_error_min',sqr_error_min)
        
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
    
    y_test_last_col = y_test.iloc[1:].tolist()
    print(len(y_test_last_col),y_test_last_col.count(0),y_test_last_col.count(1))
    P = y_test_last_col.count(1) #P=TP+FN
    N = y_test_last_col.count(0) #N=TN+FP
    
    TP = best_tpr*P
    FP = best_fpr*N
    TN = N - FP
    FN = P - TP
    
    return TP,FP,TN,FN, best_threshold

def find_best_f1_threshold(y_test, y_score):
    max_f1 = 0
    best_threshold = 0
    precision, recall, threshold = precision_recall_curve(y_test, y_score, pos_label=1)
    for fpr_item, tpr_item, threshold_item in zip(precision, recall, threshold):
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
    
    # print('**************y_test',y_test)
    y_test_last_col = y_test.iloc[1:].tolist()
    print(len(y_test_last_col),y_test_last_col.count(0),y_test_last_col.count(1))
    P = y_test_last_col.count(1) #P=TP+FN
    N = y_test_last_col.count(0) #N=TN+FP
    
    TP = best_tpr*P
    FP = best_fpr*N
    TN = N - FP
    FN = P - TP
    
    return TP,FP,TN,FN, best_threshold

crossNumber = 3
# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=crossNumber,shuffle=True, random_state=42)
# cv = ShuffleSplit(n_splits=3, test_size=0.33, random_state=0)
# model = LogisticRegression()
model = LogisticRegression(max_iter=500)
logreg = OneVsRestClassifier(model)


counter = 0
cvLabel = ['PR','ST','LT']
fig, ax = plt.subplots()
# ax1 = copy.deepcopy(ax)

for m in myY: #draw all three ROCs
#for m in myY[:1]: #draw only PR ROC
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    # fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, m)):
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('X_train size:', len(train), 'X_test size:', len(test))

        logreg.fit(X.iloc[train], m.iloc[train])
        viz = RocCurveDisplay.from_estimator(
            logreg,
            X.iloc[test],
            m.iloc[test],
            name="ROC fold {}".format(i),
            alpha=0.0,
            lw=1,
            ax=ax,
        )
        
        # feature importance
        # from sklearn.inspection import permutation_importance
        # # to get permutation: 
        # results = permutation_importance(logreg, X.iloc[train], m.iloc[train], scoring='accuracy')
        # # get important features:
        # important_features = results.importances_mean
        # # print all features importance:
        # for i,v in enumerate(important_features):
        #   print('Feature: %0d, Score: %.5f' % (i,v))
        
        # (pd.Series(logreg.feature_importances_, index=X.columns)
        # .nlargest(4)
        # .plot(kind='barh'))   
        
        my_X_train = X.iloc[train]
        my_Y_train = m.iloc[train]
        my_X_test = X.iloc[test]
        my_Y_test = m.iloc[test]
        y_pred = logreg.predict(my_X_test) 
        accuracy = accuracy_score(my_Y_test, y_pred)
        precision = precision_score(my_Y_test, y_pred)
        recall = recall_score(my_Y_test, y_pred)
        f1_score = (2 * precision * recall) / (precision + recall)
        # print(cvLabel[counter], "Accuracy:", accuracy, 'Precision', precision, 'Recall',recall, 'f1 score', f1_score)
        # cm = confusion_matrix(my_Y_test, y_pred)
        # print(cm)
        # tn, fp, fn, tp = cm.ravel()
        # print('TP:', tp, 'TN:',tn, 'FP:',fp,'FN:',fn)
        
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        y_score = logreg.fit(my_X_train, my_Y_train).decision_function(my_X_test)        
        # TP,FP,TN,FN, best_threshold = find_best_fit_2_threshold(my_Y_test, y_score)
        TP,FP,TN,FN, best_threshold = find_best_f1_threshold(my_Y_test, y_score)
        print('TP',TP,'FP',FP,'TN',TN,'FN',FN,'best threshold', best_threshold)
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        f1 = 2*precision*recall/(precision+recall)
        print(cvLabel[counter],'recall',recall,'precision',precision,'accuracy',accuracy,'f1',f1)
        
        coefficients = logreg.coef_[0]
        feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(coefficients)})
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        print(cvLabel[counter],'##################logreg feature_importance #####################')
        print(feature_importance)
        
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        
    

    colorMean = ["r","g","b"]
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color=colorMean[counter],
        label=r"%s Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (cvLabel[counter], mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    if counter == 2:
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color=colorMean[counter],
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
    else:
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color=colorMean[counter],
            alpha=0.2,
            # label=r"$\pm$ 1 std. dev.",
        )
        
    counter+=1
    
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black", label="Chance", alpha=0.8)
ax.set(
   xlim=[-0.05, 1.05],
   ylim=[-0.05, 1.05],
   title="Regression Cross Validation in PR, ST, LT",
)
ax.legend(loc="lower right",prop={'size': 8.5})
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
if hasGridline:
    plt.grid(which='major', color='#DDDDDD', linestyle=':', linewidth=0.5)
else:
    plt.grid(False)
    
# Here is the trick
plt.gcf()
handles, labels = plt.gca().get_legend_handles_labels()
index_list = [3,7,11,12,13]
handles = np.take(handles, index_list)
labels = np.take(labels, index_list)
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())


figureName = 'Figure10_regression_3.png'
print(figureName+' saved!')
plt.savefig(figureName, format="png",dpi=600)
plt.show()

