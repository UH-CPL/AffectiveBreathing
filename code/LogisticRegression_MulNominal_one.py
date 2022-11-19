# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 12:59:15 2022

@author: sunnf
"""
#https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt 

hasGridline = False

plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

csvname = 'featuresNew_12_cross.csv'

isNewFile = False
if 'cross' in csvname:
    isNewFile = True
    
data = pd.read_csv(csvname,header=0)
#log transform wav_avg

data = data.dropna()
#-1 means RB treatment, we remove RB treatments since it is baseline
data.drop(data[data['M_Stress']==-1].index, inplace = True)
#0 means the subject does not have RB treatments, thus it NORM features are 0
data.drop(data[data['BVC_NORM']== 0].index, inplace = True)
data = data.sort_values(by=['M_Stress','Subject'])

print("data is ///////////////////////")
print(data.shape)

if isNewFile == True:
    targetCol = 'M_Stress'
else:
    targetCol = 'MLStress'
    
#print columns 'y' represents 1 and 0 rows
print(data[targetCol].value_counts())
sns.countplot(x=targetCol,data=data,palette='hls')
plt.show()
plt.savefig('count_plot')

count_no_sub = len(data[data[targetCol]==0])
count_sub = len(data[data[targetCol]==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no stress is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of stress", pct_of_sub*100)

# #create dummy variables
cat_vars=['Subject','Treatment']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
#selected 8 features
if isNewFile != True:
    to_keep=['VolCycle','VolCycle_STD','VolCycle_Ratio','VolSec','VolSec_Ratio','RTQ','RTQ_STD','RTQ_Ratio','BR','BR_STD','BR_Ratio','Height','Height_STD','Height_Ratio','Length','Length_STD','Length_Ratio',targetCol]
else: 
    to_keep=['BVC_AVG','BVT_AVG','WA_AVG','BVC_SD','WA_SD','BVC_NORM','BVT_NORM','WA_NORM','BVC_SD_NORM','WA_SD_NORM','RTQ_AVG','RTQ_SD','RTQ_NORM','RTQ_SD_NORM','BR_AVG','WL_AVG','WL_SD','BR_SD','BR_NORM','WL_NORM','WL_SD_NORM','BR_SD_NORM',targetCol]

data_final=data[to_keep]

#over sample the NO data records, so the yes and no samples 
# are having balanced numbers
X = data_final.loc[:, data_final.columns != targetCol]
y = data_final.loc[:, data_final.columns == targetCol]

from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
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

data_final_vars=data_final.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]
from sklearn.preprocessing import label_binarize

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
    cols = ['WA_AVG','BVT_AVG','BVC_AVG','BVC_NORM','BVT_NORM','WA_NORM','RTQ_AVG','RTQ_NORM','WL_AVG','WL_SD','BR_NORM']
    cols = ['WA_AVG','BVC_AVG','BVT_NORM','RTQ_AVG','WL_AVG','WL_SD']
    #8
    cols = ['WA_AVG','BVC_AVG','BVT_NORM','RTQ_AVG','WL_AVG','WL_SD']
    #9
    cols = ['WA_AVG','BVT_AVG','BVT_NORM','WA_NORM','RTQ_AVG','RTQ_NORM','WL_AVG','BR_NORM']
    cols = ['BVT_NORM','RTQ_AVG','RTQ_NORM','WL_AVG','WL_SD','WA_SD']

else:
    print('$$$$$$$$$$$$$$$$$$Processing OLD file$$$$$$$$$$$$$$$$')
    #keep all
    cols=['VolCycle','VolCycle_STD','VolCycle_Ratio','VolSec','VolSec_Ratio','RTQ','RTQ_STD','RTQ_Ratio','BR','BR_Ratio','Height','Height_STD','Height_Ratio','Length','Length_STD','Length_Ratio']
 
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
    
#BVC_SD,BVC_NORM,BVT_NORM,BR_AVG,WL_AVG,BR_SD,WA_SD,WL_NORM
X=os_data_X[cols]
y=os_data_y[targetCol]

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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = LogisticRegression(max_iter=300)
logreg = OneVsRestClassifier(model)
result = logreg.fit(X_train, y_train)

y_score = logreg.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
from sklearn.metrics import classification_report

#print('123451234512345123451234512345123451234512345')

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

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

plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

if hasGridline:
    plt.grid(which='major', color='#DDDDDD', linestyle=':', linewidth=0.5)
else:
    plt.grid(False)

plt.savefig('Figure9.png', format="png",dpi=300)
plt.show()

# #############################################################################
# Classification and ROC analysis
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc

crossNumber = 3
# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=crossNumber,shuffle=True, random_state=42)
model = LogisticRegression(max_iter=150)
logreg = OneVsRestClassifier(model)

counter = 0
cvLabel = ['PR','ST','LT']
for m in myY:
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, m)):
        logreg.fit(X.iloc[train], m.iloc[train])
        viz = RocCurveDisplay.from_estimator(
            logreg,
            X.iloc[test],
            m.iloc[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        
    
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="$AR$ - " + cvLabel[counter],
    )
    ax.legend(loc="lower right",prop={'size': 8.5})
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if hasGridline:
        plt.grid(which='major', color='#DDDDDD', linestyle=':', linewidth=0.5)
    else:
        plt.grid(False)
    
    num_2_letter = ['a','b','c']
    figureName = 'Figure10'+str(num_2_letter[counter])+'.png'
    print(figureName+' saved!')
    plt.savefig(figureName, format="png",dpi=600)
    plt.show()
    counter+=1

