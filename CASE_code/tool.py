# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 19:43:08 2023

@author: sunnf
"""
import numpy as np
import math
import statistics as stat

def bandPassWithMovingAvg(data, lowcut, highcut):
    lowCutData = MovingAvg(data, lowcut)
    highCutData = MovingAvg(data, highcut)
    subtracted_array = np.subtract(highCutData, lowCutData)
    return list(subtracted_array),lowCutData, highCutData 

def MovingAvg(data, size):
    newData = []
    border = math.floor(size/2)
    # print('size:',size)
    # print('border',border)
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

def findExtremeIndices(data, indexShift, isMax, s_threshold, sampleHz):
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

#rule 1, between two min indices, must have 1 value > threshold, if not, remove the larger min point
def cleanMinPoints(data, threshold, min_indices, min_values):
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

#rule 2, between two min indices, must have at least 1 max, if < 1, remove the larger min point
def cleanMinPoints_2(min_indices, min_values, max_indices):   
    indices_2_del = []
    values_2_del = []
    
    for i in range(len(min_indices)-1):
        left = math.floor(min_indices[i])
        right = math.floor(min_indices[i+1])
        #SCAN max_indices, find if there is a max_indice between left/right
        hasMaxFlag = False
        for max_indice in max_indices:
            if max_indice > left and max_indice< right:
                hasMaxFlag = True
        #if not find any min_indice in between, must remove the smaller MAX point
        if hasMaxFlag == False:
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
    
    #delete extra max points at the beginning and the end
    #all max_points cannot be smaller than 1st min point, and 
    # larger than last min point
    #max points cannot be the 1st and the last point    
    for max_indice,max_value in zip(max_indices,max_values):
        if max_indice < min_indices[0]:
            indices_2_del.append(max_indice)
            values_2_del.append(max_value)
            
        if max_indice > min_indices[-1]:
            indices_2_del.append(max_indice)
            values_2_del.append(max_value)

    #delete extra max points in the middle of two minPoints
    for i in range(len(min_indices)-1):
        left = math.floor(min_indices[i])
        right = math.floor(min_indices[i+1])
        #find all max_points in between these two min_points
        max_inside_indices = []
        max_inside_values = []
        for j in range(len(max_indices)-1):
            if right > max_indices[j] > left:
                max_inside_indices.append(max_indices[j])
                max_inside_values.append(max_values[j])
        #search the inside max_points and only keep the max_value
        cur_max_value = 0
        cur_max_index = 0
        for j in range(len(max_inside_values)):
            if cur_max_value < max_inside_values[j]:
                cur_max_value = max_inside_values[j]
                cur_max_index = max_inside_indices[j]
        
        if len(max_inside_indices) > 1:
            #remove the max value and index
            max_inside_indices.remove(cur_max_index)
            max_inside_values.remove(cur_max_value)
            indices_2_del.extend(max_inside_indices)
            values_2_del.extend(max_inside_values)
    
    # for i in range(len(max_indices)-1):
    #     left = math.floor(max_indices[i])
    #     right = math.floor(max_indices[i+1])
    #     #SCAN min_indices, find if there is a min_indice between left/right
    #     hasMinFlag = False
    #     for min_indice in min_indices:
    #         if min_indice > left and min_indice< right:
    #             hasMinFlag = True
    #     #if not find any min_indice in between, must remove the smaller MAX point
    #     if hasMinFlag == False:
    #         if max_values[i] < max_values[i+1]:
    #             indices_2_del.append(max_indices[i])
    #             values_2_del.append(max_values[i])
    #         else:
    #             indices_2_del.append(max_indices[i+1])
    #             values_2_del.append(max_values[i+1])
    
    new_max_values = [x for x in max_values if x not in values_2_del]
    new_max_indices = [x for x in max_indices if x not in indices_2_del]
    
    return new_max_indices,new_max_values

def getVolCycle(wave,scale):
    volCycle = 0
    fourPiSquare = 4*math.pi*math.pi
    for i in range(len(wave)-1):
        volCycle+= abs(pow(wave[i]*scale/100000,3) - pow(wave[i+1]*scale/100000,3))/fourPiSquare
    return   volCycle

def getRTQ(waveSmoothed):
    inCounter = 0
    exCounter = 0
    for i in range(len(waveSmoothed)-1):
        if waveSmoothed[i] < waveSmoothed[i+1]:
            inCounter+=1
        elif waveSmoothed[i] > waveSmoothed[i+1]:
            exCounter+=1
    
    return inCounter/exCounter

import csv
def writeCvsLine(featureList, filename):
    with open(filename, 'a', newline='') as f: 
        write = csv.writer(f)
        write.writerow(featureList)
        f.close()