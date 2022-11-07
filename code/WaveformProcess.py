# -*- coding: utf-8 -*-
"""
@author: sunnf
"""  
from scipy.signal import butter, lfilter
sampleHz = 25

#input file contains breathing chest stripe length recorded per 1/25 second.
input_filename = '..\\data\\BR_Filtered_Raw.csv' 
#output file contains 18 features of all records by participant, by treatment
output_filename = 'featuresNew_12_cross.csv'
#Subject was named from T003 to T178
startIndx = 3
endIndx = 179

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import freqz
import math

def get_extreme_valuePair(inputlist, threhold): 
    #get the minimum value in the list
    res=[]
    min_value = min(inputlist) 
    max_value = max(inputlist)
    #return the index of minimum value
    if threhold<0 and min_value <= threhold:
        res = [i for i,val in enumerate(inputlist) if val==min_value]
        if len(res)==1 and res[0] > 0 and res[0]<24 : # and (res[0] > len(inputlist)/4 and res[0] < len(inputlist)*3/4):
            return res, min_value
    elif threhold>0 and max_value >= threhold:        
        res = [i for i,val in enumerate(inputlist) if val==max_value]
        if len(res)==1 and res[0] > 0 and res[0]<24:
            return res, max_value
    #failed, return empty list
    return [], 0

from itertools import islice   

import statistics as stat
def findMedianTV(data, indexShift):
    valleyValues = []
    peakValues = []
    
    for i in range(2, len(data)-2):
        if data[i-2] < data[i-1] and data[i-1] < data[i] and data[i] > data[i+1] and data[i+1] > data[i+2]:
            peakValues.append(data[i])
        elif data[i-2] > data[i-1] and data[i-1] > data[i] and data[i] < data[i+1] and data[i+1] < data[i+2]:
            valleyValues.append(data[i])
    medianPeak = stat.median(peakValues)
    medianValley = stat.median(valleyValues)
    #print(medianPeak, medianValley)
    medianTV = medianPeak - medianValley
    return medianTV
    
def findExtremeIndices(data, indexShift, isMax, s_threshold):
    s_max = max(data) - min(data)
    #print('s_threshold is:', s_threshold)
    if isMax==False:
        s_threshold = -s_threshold
    #steps to find min point in 1 sec range (25 frames)
    #1. find min value, must < - s_threshold
    #2. find its index, must be unique
    #3. Make sure it is in the middle. (not first or last point)
    ext_indices = []
    ext_values = []
    ext_index = []
    data_iter = iter(data)
    counter = 0
    for i in range(len(data)):
        if counter == 0:
            ext_index, ext_value = get_extreme_valuePair(data[i:i+sampleHz], s_threshold)
            if len(ext_index)==1:#cannot be first and last three points
                ext_indices.append((ext_index[0]+indexShift+i))
                ext_values.append(ext_value)
                counter = sampleHz
                #next(islice(data_iter, 25, 4), '')
                #i=min_index+25 #move to 1 sec later
        else:
            counter-=1
    return ext_indices, ext_values

def cleanMinPoints(data, threshold, min_indices, min_values):
    #rule 1, between two min indices, must have at least 1 max, if < 1, remove the larger min point
    
    indices_2_del = []
    values_2_del = []
    for i in range(len(min_indices)-1):
        left = math.floor(min_indices[i])
        right = math.floor(min_indices[i+1])
        #print('left:',left,left/25,'right:',right,right/25)
        max_value = max(data[left:right])
        if max_value < threshold:
            if min_values[i] > min_values[i+1]:
                indices_2_del.append(min_indices[i])
                values_2_del.append(min_values[i])
            else:
                indices_2_del.append(min_indices[i+1])
                values_2_del.append(min_values[i+1])
    
    new_min_values = [x for x in min_values if x not in values_2_del]
    new_min_indices = [x for x in min_indices if x not in indices_2_del]
    
    return new_min_indices,new_min_values

def cleanMaxPoints(min_indices, max_values, max_indices):
    #rule 1, between two max indices, must have at least 1 min, if < 1, remove the smaller max point
    
    indices_2_del = []
    values_2_del = []
    
    for i in range(len(max_indices)-1):
        left = math.floor(max_indices[i])
        right = math.floor(max_indices[i+1])
        #SCAN min_indices, find if there is a min_indice between left/right
        hasMinFlag = False
        for min_indice in min_indices:
            if min_indice > left and min_indice< right:
                hasMinFlag = True
        #if not find any min_indice in between, must remove the smaller MAX point
        if hasMinFlag == False:
            if max_values[i] < max_values[i+1]:
                indices_2_del.append(max_indices[i])
                values_2_del.append(max_values[i])
            else:
                indices_2_del.append(max_indices[i+1])
                values_2_del.append(max_values[i+1])
    
    new_max_values = [x for x in max_values if x not in values_2_del]
    new_max_indices = [x for x in max_indices if x not in indices_2_del]
    
    return new_max_indices,new_max_values

def divby10(x): return x//10000

def bandPassWithMovingAvg(data, lowcut, highcut):
    lowCutData = MovingAvg(data, lowcut)
    highCutData = MovingAvg(data, highcut)
    subtracted_array = np.subtract(highCutData, lowCutData)
    return list(subtracted_array),lowCutData, highCutData 

def MovingAvg(data, size):
    newData = []
    border = math.floor(size/2)
    print('size:',size)
    print('border',border)
    for i in range(border):
        newData.append(0)
    for i in range(border, len(data)-border):
        total = 0
        counter=0
        for j in range(-border+i, border+i):
            total+= data[j]
            counter+=1
        newData.append(total/counter)
    for i in range(border):
        newData.append(0)
    return newData

import itertools as it

def getTreatment(queryString):
    treaments = ['RB','ST','DT','PR']
    for tr in treaments:
        if tr in queryString:
            return tr
    
def getVolCycle(wave):
    volCycle = 0
    fourPiSquare = 4*math.pi*math.pi
    for i in range(len(wave)-1):
        volCycle+= abs(pow(wave[i]/100000,3) - pow(wave[i+1]/100000,3))/fourPiSquare
    return   volCycle

import csv
def writeCvsLine(featureList, isFileExists):
    header = ['Subject','WA_AVG','BVT_AVG','BVC_AVG','BVC_SD','BVT_SD','WA_SD','BVC_NORM','BVC_SD_NORM','BVT_NORM','WA_NORM','RTQ_AVG','RTQ_SD','RTQ_NORM','RTQ_SD_NORM','BR_AVG','WL_AVG','WL_SD','BR_SD','BR_NORM','BR_SD_NORM','WA_SD_NORM','WL_NORM','WL_SD_NORM','Treatment','M_Stress']
    with open(output_filename, 'a', newline='') as f:
        write = csv.writer(f)
        if isFileExists != True:
            print('write csv header')
            write.writerow(header)
        write.writerow(featureList)
        f.close()

def getMeanWithoutMaxMin(valueList):
    if len(valueList)-2 <=0:
        return sum(valueList)/len(valueList)
    else:
        return (sum(valueList) - min(valueList) - max(valueList))/(len(valueList)-2)

def getRTQ(waveSmoothed):
    inCounter = 0
    exCounter = 0
    for i in range(len(waveSmoothed)-1):
        if waveSmoothed[i] < waveSmoothed[i+1]:
            inCounter+=1
        elif waveSmoothed[i] > waveSmoothed[i+1]:
            exCounter+=1
    
    return inCounter/exCounter
        
import sys
   
def main(filename, subject, queryString, baseDict):
    min_Threshold = sys.maxsize
    csv_file = filename
    tr = getTreatment(queryString)
    gen = pd.read_csv(csv_file, chunksize=10000000)
    data = pd.concat((x.query(queryString) for x in gen), ignore_index=True)
    if data.shape[0] == 0:
        print(subject, tr, ' does not exist!')
        return
    print("data is ///////////////////////")
    print(data.shape)
    print(list(data.columns))

    waveform_list = data['BreathingWaveform'].tolist()
    # Sample rate and desired cutoff frequencies (in Hz).
    lowcut = 0.1 #1Hz, 25
    highcut = 1.0   #

    nsamples = len(waveform_list)
    t = np.arange(0, nsamples) / sampleHz
    highCutSampleSize = 25 # 1hz
    lowCutSampleSize = 251 # 0.1hz
    startIndex = lowCutSampleSize
    endIndex = nsamples - lowCutSampleSize
    
    #apply bandpass filter
    bandPassWaveform = butter_bandpass_filter(waveform_list, lowcut, highcut, sampleHz, order=1)
    myBandPassWaveform, lowCutData, highCutData = bandPassWithMovingAvg(waveform_list, lowCutSampleSize, highCutSampleSize)
    #remove the head and end wasted values
    new_myBandPassWaveform = myBandPassWaveform[startIndex:endIndex]
    if 'RB' in queryString:
        medianTV = findMedianTV(new_myBandPassWaveform,startIndex)
        min_Threshold = min(medianTV/th_scale,min_Threshold)
        baseDict['s_threshold'] = min_Threshold
    if 'ST' in queryString:
        medianTV = findMedianTV(new_myBandPassWaveform,startIndex)
        min_Threshold = min(medianTV/th_scale,min_Threshold)
        baseDict['s_threshold'] = min_Threshold
 
    minIndices, minValues = findExtremeIndices(new_myBandPassWaveform, startIndex, False, baseDict['s_threshold'])
    maxIndices, maxValues = findExtremeIndices(new_myBandPassWaveform, startIndex, True, baseDict['s_threshold'])

    ####now clean up the max and min indices
    minIndices, minValues = cleanMinPoints(myBandPassWaveform, baseDict['s_threshold'], minIndices, minValues)
    maxIndices, maxValues = cleanMaxPoints(minIndices, maxValues, maxIndices)

    minIndicesTimestamp = [x / sampleHz for x in minIndices]
    maxIndicesTimestamp = [x / sampleHz for x in maxIndices]
    plt.figure()
    #convert Y-Axis from units to cm
    scale = 100000
    timeScale = 1
    newList = [x / scale for x in myBandPassWaveform]
    timeStamp = [x / timeScale for x in t]
    plt.plot( timeStamp[startIndex:endIndex],newList[startIndex:endIndex],label='Filtered Signal')
    plt.ylim([-0.8, 1.0])
    newList = [x / scale for x in minValues]
    timeStamp = [x / timeScale for x in minIndicesTimestamp]
    plt.plot(timeStamp[0:len(newList)],newList, linestyle=' ', marker='o', color='b', label='Valley Markers')
    newList = [x / scale for x in maxValues]
    timeStamp = [x / timeScale for x in maxIndicesTimestamp]
    plt.plot(timeStamp,newList, linestyle=' ', marker='o', color='r', label='Peak Markers')
    plt.xlabel("Time [s]")
    plt.ylabel("Normalized Circumference [cm]")
    plt.axhline(0, color='black', linewidth=1)
    plt.axhline(baseDict['s_threshold']/scale, color='green', linestyle='dashed', linewidth=1, label=('Threshold ='+'\u00B1' + str(baseDict['s_threshold']/scale)[0:5]+'cm'))
    plt.axhline(-baseDict['s_threshold']/scale, color='green',linestyle='dashed', linewidth=1)
    plt.legend(loc="best")
    title = 'Participant '+subject+' Breathing Cycle Detection - '+tr+' Session'
    if tr == 'DT':
        imgname = subject+'-LT'
    else:
        imgname = subject+'-'+tr
    plt.title(title)
    plt.savefig('CycleDetection_'+imgname+'_a.png', format='png',dpi=300)
    print('save fig:' + 'CycleDetection_'+imgname+'_a.png')
    plt.show()
    
    waveMatrix = []
    maxLength = 0
    width = 5.8
    height = 15
    #figure for new waveform plots
    fig = plt.figure(figsize=(width,height))
    #change font size bigger
    plt.rcParams.update({'font.size': 18})
    
    #features calculation
    #volumes on original signals (wave)
    VolCycleList = []
    VolCycle = 0
    VolCycle_STD = 0
    VolCycle_Ratio = 0
    VolCycleSTD_Ratio = 0
    
    VolSecList = []
    VolSec = 0
    VolSec_STD = 0
    VolSec_Ratio = 0
    
    #rest on filtered signals (waveSmoothed)
    RTQList = []
    RTQ = 0    
    RTQ_STD = 0
    RTQ_Ratio =0
    RTQSTD_Ratio=0
    
    BRList = []
    BR = 0
    BR_STD = 0
    BR_Ratio =0
    BRSTD_Ratio=0
    
    HeightList = []
    Height = 0
    Height_STD = 0
    Height_Ratio = 0
    HeightSTD_Ratio = 0
    
    LengthList = []
    Length = 0
    Length_STD = 0
    Length_Ratio = 0
    LengthSTD_Ratio = 0
    
    for i in range(int(len(minIndices)/2)):
        #normalized wave is used for VolCycle
        wave = waveform_list[minIndices[i]:minIndices[i+1]]
        volume_cycle = getVolCycle(wave)
        VolCycleList.append(volume_cycle)
        VolSecList.append(volume_cycle/(len(wave)/sampleHz))#calculate volume/sec
        
        #waveSmoothed is used for RTQ/BR/Height/Length
        waveSmoothed = myBandPassWaveform[minIndices[i]:minIndices[i+1]]
        RTQ = getRTQ(waveSmoothed)
        RTQList.append(RTQ)
        BRList.append(60/(len(waveSmoothed)/sampleHz))
        HeightList.append(max(waveSmoothed) - min(waveSmoothed))
        LengthList.append(len(waveSmoothed))
        
        maxLength =max(maxLength, minIndices[i+1] - minIndices[i])
        waveMatrix.append(wave)
        wave =  [x + 3000*i for x in wave]
        x = [i/25 for i in range(len(wave))]
        y = [i/100000 for i in wave]
        plt.plot(x, y, label='Waveform' + str(i))
    
    #feature calculation
    VolCycle = getMeanWithoutMaxMin(VolCycleList)
    VolCycle_STD = stat.pstdev(VolCycleList) 

    VolSec = getMeanWithoutMaxMin(VolSecList)
    VolSec_STD = stat.pstdev(VolSecList) 
        
    RTQ = getMeanWithoutMaxMin(RTQList)
    RTQ_STD = stat.pstdev(RTQList)
    
    BR = getMeanWithoutMaxMin(BRList)    
    BR_STD = stat.pstdev(BRList)
        
    Height = getMeanWithoutMaxMin(HeightList)
    Height_STD = stat.pstdev(HeightList)
    
    Length = getMeanWithoutMaxMin(LengthList)
    
    Length_STD = stat.pstdev(LengthList)
    
    #normalization calculation
    if baseDict['BR_base']!=0:
        RTQ_Ratio = RTQ/baseDict['RTQ_base']
        BR_Ratio = BR/baseDict['BR_base']
        Height_Ratio = Height/baseDict['Height_base']
        Length_Ratio = Length/baseDict['Length_base']
        VolCycle_Ratio = VolCycle/baseDict['VolCycle_base']
        VolSec_Ratio = VolSec/baseDict['VolSec_base']
        
        RTQSTD_Ratio = RTQ_STD/stdDict['RTQSTD_base']
        BRSTD_Ratio = BR_STD/stdDict['BRSTD_base']
        HeightSTD_Ratio = Height_STD/stdDict['HeightSTD_base']
        LengthSTD_Ratio = Length_STD/stdDict['LengthSTD_base']
        VolCycleSTD_Ratio = VolCycle_STD/stdDict['VolCycleSTD_base']

    #store RB features for normalization calculation 
    if 'RB'==tr:
        baseDict['RTQ_base'] = RTQ
        baseDict['BR_base'] = BR
        baseDict['Height_base'] = Height
        baseDict['Length_base'] = Length
        baseDict['VolCycle_base'] = VolCycle
        baseDict['VolSec_base'] = VolSec
        
        stdDict['RTQSTD_base'] = RTQ_STD
        stdDict['BRSTD_base'] = BR_STD
        stdDict['HeightSTD_base'] = Height_STD
        stdDict['LengthSTD_base'] = Length_STD
        stdDict['VolCycleSTD_base'] = VolCycle_STD
        stdDict['VolSec_STD_base'] = VolCycle_STD

    label_size = 22
    plt.xlabel("Time [s]", fontsize=label_size)
    plt.ylabel("Rib Cage [cm]",fontsize=label_size)    
    plt.title(imgname+' Waveform', fontsize=label_size)
    plt.savefig('CycleDetection-'+imgname+'_b.png', format='png',dpi=290)
    print('save fig:' + 'CycleDetection_'+imgname+'_b.png')
    plt.show()
    plt.clf()
    
    MLStress = -1
    if tr=='ST':
        MLStress = 1
    elif tr=='DT':
        MLStress = 2
    elif tr=='PR':
        MLStress = 0

    features = [subject, Height, VolSec, VolCycle, VolCycle_STD,VolSec_STD, Height_STD, VolCycle_Ratio, VolCycleSTD_Ratio,
                VolSec_Ratio, Height_Ratio,  RTQ, RTQ_STD,RTQ_Ratio,RTQSTD_Ratio,
                BR,Length, Length_STD, BR_STD, BR_Ratio, BRSTD_Ratio, HeightSTD_Ratio, Length_Ratio, LengthSTD_Ratio,
                tr, MLStress]
    
    isFileExist = exists(output_filename)
    writeCvsLine(features,isFileExist)
    del data

subjects = []
#Subject was named from T003 to T178
for i in range(startIndx, endIndx):
    numStr = str(i)
    subjectStr = ''
    if len(numStr) == 1:
        subjectStr = 'T00'+numStr
    elif len(numStr) == 2:
        subjectStr = 'T0'+numStr
    elif len(numStr) == 3: 
        subjectStr = 'T'+numStr
    subjects.append(subjectStr)

print(subjects)
th_scale = 5

from os.path import exists
#process one subject a time
for s in subjects:       
    #check if a file exists
    if exists(input_filename)!=True: 
        print(s, 'does not exist!')
        continue
    query=[]
    query.append("Participant_ID == '"+s+"' & Treatment=='RB'")
    query.append("Participant_ID == '"+s+"' & Treatment=='ST'")
    query.append("Participant_ID == '"+s+"' & Treatment=='DT'")
    query.append("Participant_ID == '"+s+"' & Treatment=='PR'")
    
    baseDict = {'VolCycle_base':0, 'VolSec_base':0,'RTQ_base':0,
                'BR_base':0,'Height_base':0,'Length_base':0, 's_threshold':0}
    stdDict = {'VolCycleSTD_base':0, 'VolSecSTD_base':0,'RTQSTD_base':0,
                'BRSTD_base':0,'HeightSTD_base':0,'LengthSTD_base':0}
    
    for q in query:
        main(input_filename, s, q, baseDict)