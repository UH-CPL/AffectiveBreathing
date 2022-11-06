# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:25:53 2022

@author: sunnf
"""
isRaw = True
isML2 = True
#importing libraries
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
#Loading the feature dataset
#generate the cross correlation matrix Figure
inputfilename = 'featuresNew_11_cross.csv' 
outputImagename = 'Figure8'                             
df = pd.read_csv(inputfilename,header=0)
df = df.dropna()
#-1 means RB treatment, we remove RB treatments since it is baseline
df.drop(df[df['M_Stress']==-1].index, inplace = True)
#0 means the subject does not have RB treatments, thus it NORM features are 0
df.drop(df[df['BVC_NORM']== 0].index, inplace = True)

print("data is ///////////////////////")
print(df.shape)
print(list(df.columns))

if 'features' not in inputfilename:
#cols=['VolSec','RTQ','RTQ_STD','BR','Height STD','Length']
    df = df.drop(['PR_Stress','ST_Stress','DT_Stress','M_Stress'],1)
    df = df.drop(['Mean','Median','Std','Skewness','Mean STD','Median_STD','STD_STD','Skewness_STD'], 1)
else:
    df = df.drop(['Subject', 'M_Stress'],1)
    df = df.drop(['BVC_SD_NORM','WA_SD_NORM', 'RTQ_SD_NORM','WL_SD_NORM','BR_SD_NORM'],1)
    
#df = df.drop(['BVC_AVG', 'BVC_NORM','VolCycle_STD','VolCycle_Ratio','Height','Height_Ratio','Length','Length_Ratio','Length_STD'],1)
if isRaw != True:
    if isML2 == False:
        df = df.drop(['BVC-AVG','BVC-SD','BVT-NORM','WA-AVG','WA-NORM','WL-AVG','WL-SD','WL-NORM'], 1)
    else:
        df = df.drop(['BVC_AVG','BVC_SD','BVT_NORM','WA_AVG','WA_NORM','WL_AVG','WL_SD','WL_NORM'], 1)

#Using Pearson Correlation
if isRaw == True:
    fig, ax = plt.subplots(figsize=(12,10),dpi=300)
    #plt.figure(figsize=(12,10),dpi=300)
else:
    plt.figure(figsize=(7.2,6),dpi=300)
cor = df.corr()
print(cor)
g = sns.heatmap(cor, annot=True, cmap=plt.cm.Reds, fmt='.2f')
plt.xticks(rotation=45)
plt.yticks(rotation=45)


import matplotlib.patches as patches
def drawPatch(xy, size, borderColor,borderWidth, fillFlag, ax):
    drawPatchOffDiag(xy, xy, size, borderColor,borderWidth, fillFlag, ax)

def drawPatchOffDiag(x,y, size, borderColor,borderWidth, fillFlag, ax):
    ax.add_patch(
     patches.Rectangle(
         (x, y),
         size,
         size,
         edgecolor=borderColor,
         fill=fillFlag,
         lw=borderWidth
     ) )
    
borderColor = '#00EE00'
borderWidth = 3
fillFlag = False
if isRaw == True:
    drawPatch(0,3,borderColor,borderWidth,fillFlag,ax)
    drawPatch(2,2,borderColor,borderWidth,fillFlag,ax)
    drawPatch(4,2,borderColor,borderWidth,fillFlag,ax)
    drawPatch(6,2,borderColor,borderWidth,fillFlag,ax)
    drawPatch(12,2,borderColor,borderWidth,fillFlag,ax)
    drawPatch(16,2,borderColor,borderWidth,fillFlag,ax)
    plt.savefig(outputImagename+'.png', format="png",dpi=600)
else:
    plt.savefig(outputImagename+'.eps', format="eps",dpi=600)
plt.show()