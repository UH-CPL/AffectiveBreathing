# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:50:29 2023

@author: sunnf
"""
import tool
import pandas as pd
import sys
import pickle
import statistics as stat
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter

##################################################################
# Video-label	Video-ID	Duration (in ms)	Source (Year)
# 			
# amusing-1	1	185583.333	Hangover (2009)
# amusing-2	2	173083.333	When Harry Met Sally (1989)
# bluVid	11	120000	n.a. -- blue screen
# boring-1	3	118666.667	Europe Travel Skills (2013) 
# boring-2	4	160125	Matcha: The way of Tea (2012)
# endVid	12	120000	n.a. -- blue screen
# relaxed-1	5	145125	Relaxing Music with Beach (2011)
# relaxed-2	6	146750	Natural World: Zambezi (2012)
# scary-1	7	197000	Shutter (2004)
# scary-2	8	143750	Mama (2008)
# startVid	10	101500	Great Barrier Reef (2012)

#We will use video 5,6 as baseline video. relaxed-1,2


#hard code, 1/5 of TV
sampleHz = 100
th_scale = 0.1
min_Threshold = sys.maxsize
results_csv = 'caseNew_5_cross.csv'

baseDict = {'VolCycle_base':0, 'VolSec_base':0,'RTQ_base':0,
                'BR_base':0,'Height_base':0,'Length_base':0, 's_threshold':0}
stdDict = {'VolCycleSTD_base':0, 'VolSecSTD_base':0,'RTQSTD_base':0,
                'BRSTD_base':0,'HeightSTD_base':0,'LengthSTD_base':0}

waveCycleDist = 100

def plotCycleFigureNew(imgname, minIndices, waveform_list, myBandPassWaveform,VolCycleList,VolSecList,RTQList,BRList,HeightList,LengthList, maxLength,waveMatrix,maxIndices,scale):
    width = 6
    height = 15
    #start new cycles figure 
    plt.figure(figsize=(width,height))
    
    print('BR cycles:',len(minIndices)-1)
    for i in range(len(minIndices)-1):
        #normalized wave is used for VolCycle
        wave = waveform_list[minIndices[i]:minIndices[i+1]]
        wave1= myBandPassWaveform[minIndices[i]:minIndices[i+1]]
        ## hardcode
        scale = 1000
        volume_cycle = tool.getVolCycle(wave, scale)        
        VolCycleList.append(volume_cycle)
        VolSecList.append(volume_cycle/(len(wave)/sampleHz))#calculate volume/sec
        
        #waveSmoothed is used for RTQ/BR/Height/Length
        waveSmoothed = myBandPassWaveform[minIndices[i]:minIndices[i+1]]
        RTQ = tool.getRTQ(waveSmoothed)
        RTQList.append(RTQ)
        BRList.append(60/(len(waveSmoothed)/sampleHz))
        HeightList.append(max(waveSmoothed) - min(waveSmoothed))
        LengthList.append(len(waveSmoothed))
        
        maxLength =max(maxLength, minIndices[i+1] - minIndices[i])
        waveMatrix.append(wave)
        wave =  [x + waveCycleDist*i for x in wave]
        wave1 =  [x - 4000 + waveCycleDist*i for x in wave1]
        # print(wave1[::2])
        # print(len(wave1))
        # break
        plt.plot(wave, label='Waveform' + str(i))
        plt.plot(wave1, label='Smooth_Waveform' + str(i))
        
        #draw peak marker of raw/smooth signals        
        peak_x = maxIndices[i] - minIndices[i]
        if peak_x < len(wave):
            peak_y_raw = wave[peak_x]
            peak_y_smooth = wave1[peak_x]
            plt.plot(peak_x,peak_y_raw, linestyle=' ', marker='o', color='r', label='Peak Markers')
            plt.plot(peak_x,peak_y_smooth, linestyle=' ', marker='o', color='r', label='Peak Markers')

    #end cycle for loop
    plt.xlabel("Time [1/25s]")
    plt.ylabel("Rib Cage [um]")
    plt.grid(False)
    
    plt.title(imgname+' Waveform plots')
    plt.savefig('CycleDetection_'+imgname+'_b.png', format='png',dpi=300)
    print('save fig:' + 'CycleDetection_'+imgname+'_b.png')
    plt.show()

def plotCycleFigure(sub_no, video_no,imgname, minIndices, waveform_list, VolCycleList, VolSecList, RTQList,BRList,HeightList,LengthList,maxLength,waveMatrix, myBandPassWaveform,scale):
    width = 6.2
    height = 15
    #start new cycles figure     
    plt.figure(figsize=(width,height))
    #adjust margins
    plt.subplots_adjust(left=0.2)
    title_size = 22
    tick_size = 18
    
    for i in range(int(len(minIndices)/2)):
        #normalized wave is used for VolCycle
        wave = waveform_list[minIndices[i]:minIndices[i+1]]
        volume_cycle = tool.getVolCycle(wave,scale)
        VolCycleList.append(volume_cycle)
        VolSecList.append(volume_cycle/(len(wave)/sampleHz))#calculate volume/sec
        
        #waveSmoothed is used for RTQ/BR/Height/Length
        waveSmoothed = myBandPassWaveform[minIndices[i]:minIndices[i+1]]
        RTQ = tool.getRTQ(waveSmoothed)
        RTQList.append(RTQ)
        BRList.append(60/(len(waveSmoothed)/sampleHz))
        HeightList.append(max(waveSmoothed) - min(waveSmoothed))
        LengthList.append(len(waveSmoothed))
        
        maxLength =max(maxLength, minIndices[i+1] - minIndices[i])
        waveMatrix.append(wave)
        wave =  [x + waveCycleDist*i for x in wave]
        um_cm_scale = 1000
        waveSmoothed =  [71000/um_cm_scale + x/um_cm_scale + waveCycleDist*i/um_cm_scale for x in waveSmoothed]
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.gca().set_ylim(bottom=70.0,top=74.0)
        plt.gca().set_xlim(left=0,right=800)
        plt.xticks(np.arange(0, 900, 100), fontsize=tick_size,color='w')       
        plt.plot(waveSmoothed, label='Waveform' + str(i))
    
    
    plt.xlabel("Time [s]", fontsize = title_size)
    plt.ylabel("Rib Cage [cm]", fontsize = title_size)
    plt.grid(False)
    
    #set default video type to Non-scary
    video_type = 'Non-Scary'
    if video_no in [5,6]:
        video_type = 'Baseline'
    elif video_no in [7,8]:
        video_type = 'Scary'
    
    plt.title('Participant ' + str(sub_no) +' ' + video_type+' Cycles', fontsize=title_size, color='w')
        
    plt.savefig('CycleDetection_'+imgname+'_b.png', format='png',dpi=300)
    print('save fig:' + 'CycleDetection_'+imgname+'_b.png')
    plt.show()
    

def process(data, sub_no, video_no):
    global th_scale
    global min_Threshold
    
    # Sample rate and desired cutoff frequencies (in Hz).
    lowcut = 0.1
    highcut = 1.0
    if isinstance(data, (list)):
        print('data is a list')
    elif isinstance(data, (pd.Series)):
        print('data is a pandas dataframe')
        
    waveform_list = data
    nsamples = len(waveform_list)
    print(nsamples)
    t = np.arange(0, nsamples) / sampleHz #video length in secs
    #hard code, highCutSampleSize
    highCutSampleSize = 0.2*sampleHz # 1hz
    lowCutSampleSize = 10*sampleHz # 0.1hz
    startIndex = lowCutSampleSize
    endIndex = nsamples - lowCutSampleSize
    
    myBandPassWaveform, lowCutData, highCutData = tool.bandPassWithMovingAvg(waveform_list, lowCutSampleSize, highCutSampleSize)
    #remove the head and end wasted values
    new_myBandPassWaveform = myBandPassWaveform[startIndex:endIndex]
    video_length = (endIndex - startIndex)/(sampleHz*60) #video_length in min
    print('endIndex', endIndex,'startIndex', startIndex )
    #################################################################
    ## start process, find Median TV, and threshold
    medianTV = tool.findMedianTV(new_myBandPassWaveform,startIndex)
    min_Threshold = min(medianTV/th_scale,min_Threshold)
    baseDict['s_threshold'] = min_Threshold
    
    minIndices, minValues = tool.findExtremeIndices(new_myBandPassWaveform, startIndex, False, baseDict['s_threshold'], sampleHz)
    maxIndices, maxValues = tool.findExtremeIndices(new_myBandPassWaveform, startIndex, True, baseDict['s_threshold'], sampleHz)
    
    ####now clean up the max and min indices
    minIndices, minValues = tool.cleanMinPoints(myBandPassWaveform, baseDict['s_threshold'], minIndices, minValues)
    maxIndices, maxValues = tool.cleanMaxPoints(minIndices, maxValues, maxIndices)
    
    max_point_length = len(maxIndices)
    min_point_length = len(minIndices)
    
    count = 0
    while min_point_length - max_point_length > 1:
        minIndices, minValues = tool.cleanMinPoints_2(minIndices, minValues, maxIndices)
        min_point_length = len(minIndices)
        count+=1
        print('remove extra valley points!',count)
        if count > 5:
            break
    
    count = 0
    while min_point_length - max_point_length ==0:
        maxIndices, maxValues = tool.cleanMaxPoints(minIndices, maxValues, maxIndices)
        max_point_length = len(maxIndices)
        count+=1
        print('remove extra valley points!',count)
        if count > 5:
            break

    minIndicesTimestamp = [x / sampleHz for x in minIndices]
    maxIndicesTimestamp = [x / sampleHz for x in maxIndices]
    plt.figure()
    
    #convert Y-Axis from units to cm
    scale = 2000
    timeScale = 1
    newList = [x / scale for x in myBandPassWaveform]
    rawList = [(x+2.5*scale) / (scale*1.5) for x in waveform_list]
    
    timeStamp = [x / timeScale for x in t]
    plt.plot( timeStamp[startIndex:endIndex],newList[startIndex:endIndex],label='Filtered Signal')
    plt.plot( timeStamp[startIndex:endIndex],rawList[startIndex:endIndex],label='Raw Signal', linewidth=0.3)
    
    plt.ylim([-0.8, 1.0])
    newList = [x / scale for x in minValues]
    timeStamp = [x / timeScale for x in minIndicesTimestamp]
    plt.plot(timeStamp,newList, linestyle=' ', marker='o', color='b', label='Valley Markers')
    print('Valley markers#:', len(newList))
    newList = [x / scale for x in maxValues]
    timeStamp = [x / timeScale for x in maxIndicesTimestamp]
    plt.plot(timeStamp,newList, linestyle=' ', marker='o', color='r', label='Peak Markers')
    print('Peak marker#:', len(newList))
    plt.xlabel("Time [s]")
    plt.ylabel("Normalized Circumference [cm]")
    plt.axhline(0, color='black', linewidth=1)
    plt.axhline(baseDict['s_threshold']/scale, color='green', linestyle='dashed', linewidth=1, label=('Threshold ='+'\u00B1' + str(baseDict['s_threshold']/scale)[0:5]+'cm'))
    plt.axhline(-baseDict['s_threshold']/scale, color='green',linestyle='dashed', linewidth=1)
    plt.legend(loc="best")
    title = 'Participant '+str(sub_no)+' Breathing Cycle Detection - '+str(video_no)+' Session'
    imgname = str(sub_no)+'_'+str(video_no)
    plt.title(title)
    plt.savefig('CycleDetection_'+imgname+'_a.png', format='png',dpi=300)
    print('save fig:' + 'CycleDetection_'+imgname+'_a.png')
    plt.show()
    
    ############# Start waveform plots #############
    waveMatrix = []
    maxLength = 0    
    
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
    
    plotCycleFigure(sub_no, video_no,imgname, minIndices, waveform_list, VolCycleList, VolSecList, RTQList,BRList,HeightList,LengthList,maxLength,waveMatrix, myBandPassWaveform,scale)
    #feature calculation
    VolCycle = stat.mean(VolCycleList)
    VolCycle_STD = stat.pstdev(VolCycleList) 

    VolSec = stat.mean(VolSecList)
    VolSec_STD = stat.pstdev(VolSecList) 
        
    RTQ = stat.mean(RTQList)
    RTQ_STD = stat.pstdev(RTQList)
    
    BR = stat.mean(BRList)    
    BR_STD = stat.pstdev(BRList)
        
    Height = stat.mean(HeightList)
    Height_STD = stat.pstdev(HeightList)
    
    Length = (stat.mean(LengthList))*(25/sampleHz)
    
    Length_STD = stat.pstdev(LengthList)
    
    if video_no not in [5,6]:
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
    #store video 5,6 features for normalization calculation 
    else:
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

    if video_no in [1,2,3,4]:
         MLStress = 0
    elif video_no in [7,8]:
         MLStress = 1
    elif video_no in [5,6]:
         MLStress = 2
    else:
        MLStress = 3
        

    features = [sub_no, video_no, video_length, Height, VolSec, VolCycle, VolCycle_STD, VolSec_STD, Height_STD, VolCycle_Ratio, VolCycleSTD_Ratio,
                VolSec_Ratio, Height_Ratio,  RTQ, RTQ_STD,RTQ_Ratio,RTQSTD_Ratio,
                BR,Length, Length_STD, BR_STD, BR_Ratio, BRSTD_Ratio, HeightSTD_Ratio, Length_Ratio, LengthSTD_Ratio,
                MLStress]

    tool.writeCvsLine(features, results_csv)
    del data
    
######################################################################################
## main section
#only process needed videos, 5,6 are baseline video and should be processed first
video_list = [5,6,1,2,3,4,7,8]
sub_list = list(range(1,31))
# sub_list=[1]

for sub_no in sub_list:
    #load breathing data by subject
    filename = '../data/interpolated/physiological/' + 'sub_'+str(sub_no)+'.csv'
    print('processing sub:',filename)
    data = pd.read_csv(filename, sep=',', usecols=[4,9])
    
    for v_no in video_list:    
        dataFrame = data[(data['video'] == v_no)]
        print('processing sub:',filename, 'video:', v_no)
        breathing = dataFrame['rsp'].tolist()[0::10]
        new_breathing = [i * 1000 for i in breathing]
        #for video 7,8, scary videos, only take records when scary scene happen
        if v_no==7 or v_no==8:
            new_breathing = new_breathing[9000::]
        process(new_breathing, sub_no, v_no)