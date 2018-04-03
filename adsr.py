# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 14:08:34 2018

@author: stanley
"""
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import argrelextrema
#plt.grid(True)

def calVolumeDB(waveData, frameSize, overLap):
    wlen = len(waveData)
    step = frameSize - overLap
    frameNum = int(math.ceil(wlen*1.0/step))
    volume = np.zeros((frameNum,1))
    for i in range(frameNum):
        curFrame = waveData[np.arange(i*step,min(i*step+frameSize,wlen))]
        curFrame = curFrame - np.mean(curFrame) # zero-justifie
        if np.sum(curFrame*curFrame) == 0.0:
            volume[i] = 10*np.log10(np.sum(curFrame*curFrame)+0.000000000001)
        else:
            volume[i] = 10*np.log10(np.sum(curFrame*curFrame))
    return volume

####讀取wavedata(正規化)
wavepath = '../10violin/wave/' 
midipath = '../10violin/midi/' 
outputpath = 'TLS_result/'
song = '08-6'
sampFreq, snd = wavfile.read(wavepath+song+'.wav')
waveData = snd*1.0/max(abs(snd))#(snd-min(snd))/(max(snd)-min(snd))#snd*1.0/max(abs(snd))
                                #(snd-np.mean(snd))/(max(snd)-min(snd))
frameSize = 256
overLap = 128
step = frameSize - overLap

##讀取ground truth與小節音符個數
import xml.etree.cElementTree as ET
tree = ET.ElementTree(file=midipath+song[0:2]+'_violin.xml')
#for elem in tree.iterfind('part/measure'):
#    print (elem.tag, elem.attrib)
duration_dict = {'quarter':1,
                 'eighth':0.5,
                 'half':2,
                 '32nd':0.125,
                 '16th':0.25}
notes_duration = []
notes = []
Bar = []    
note_cnt = 0
is_tied = False
is_rest = False
tieds = []
for elem in tree.iter():
    if elem.tag =='rest':
        is_rest = True
        note_cnt -=1
        notes.append(False)
    #print( elem.tag, elem.attrib,elem.text)
    if elem.tag =='measure':
        Bar.append(note_cnt)
        note_cnt = 0
    if elem.tag == 'note':
        note_cnt += 1
        notes.append(True)
    if elem.tag =='tied' and elem.attrib['type'] =='start':
        tieds.append([len(notes_duration)-1,notes_duration[-1]])
        is_tied = True
    if elem.tag =='tied' and elem.attrib['type'] =='stop':
        note_cnt-=1
       # print(elem.attrib['type'])
        is_tied = False
    if elem.tag =='dot':
       # print("/////")
        notes_duration[-1]*=1.5
    if elem.tag =='type':
        if is_tied:
            notes_duration[-1] +=duration_dict[elem.text]
            tieds[-1].append(duration_dict[elem.text])
        elif is_rest:
            notes_duration[-1] +=duration_dict[elem.text]
            is_rest = False
        else: 
           # print(len(notes_duration))
            notes_duration.append(duration_dict[elem.text])
    #if elem.tag =='beats':
        #print(elem.attrib,elem.text)
    
Bar.append(note_cnt)
del(Bar[0])
time_value = notes_duration




import re
onsetpath = song+'_ground.txt'
f2 = open(wavepath+song+'_ground.txt','r')
fr2=f2.read()
fr2 = re.split('\n|\t',fr2)
fr2.pop()
if('ground' in onsetpath):
     onsets = [float(fr2[i])  for i in range(len(fr2)) if i%5==0 ]
     offsets = [float(fr2[i])  for i in range(len(fr2)) if i%5==1]
else:
     onsets = [float(fr2[i])  for i in range(len(fr2)) if i%6==2 ]
     offsets = [float(fr2[i])  for i in range(len(fr2)) if i%6==3]


#plt.show()

#v = calVolumeDB(waveData,frameSize,overLap)
#plt.plot(v)
#plt.show()

def getEnvelope(inputSignal):
# Taking the absolute value

    absoluteSignal = []
    for sample in inputSignal:
        absoluteSignal.append (abs (sample))

    # Peak detection

    intervalLength = 35 # change this number depending on your Signal frequency content and time scale
    outputSignal = []

    for baseIndex in range (0, len (absoluteSignal)):
        maximum = 0
        for lookbackIndex in range (intervalLength):
            maximum = max (absoluteSignal [baseIndex - lookbackIndex], maximum)
        outputSignal.append (maximum)

    return outputSignal
v = []
v = getEnvelope(waveData[:len(waveData)//4])
#plt.plot(v)

from scipy.signal import butter, lfilter
b,a = butter(3,150/(44100/2),'low')
v = lfilter(b,a,v)
#plt.plot(v)
#plt.show()


for i in range(10):#len(onsets)):
    print(i)
    plt.plot(waveData[int(onsets[i]*sampFreq):int(offsets[i]*sampFreq)])
    plt.show()
    temp = v[int(onsets[i]*sampFreq):int(offsets[i]*sampFreq)]
    plt.plot(temp)
    
    localmin = argrelextrema( temp, np.less)
    localmin = localmin[0].tolist()
    lminval =[]
    for j in localmin:
        lminval.append(temp[j])
    plt.plot(localmin,lminval)
    
    left = int(len(lminval)/5)
    right = int((len(lminval)/5)*4)
    leftmin = (min(lminval[:left]))
    rightmin = (min(lminval[right:]))
    #plt.plot(localmin[lminval.index(leftmin)],leftmin,'*')
    
    attack = max(temp[localmin[lminval.index(leftmin)]:localmin[lminval.index(rightmin)]])
    attackindex,= np.where(temp == attack)  #, = temp.index(attack)  
    plt.plot([localmin[lminval.index(leftmin)],attackindex,localmin[lminval.index(rightmin)]],
                       [leftmin,attack,rightmin])
    plt.show()
"""
v = calVolumeDB(waveData,frameSize,overLap)
for i in range(len(onsets)):
    print(i)
    plt.plot(waveData[int(onsets[i]*sampFreq):int(offsets[i]*sampFreq)])
    plt.show()
    temp = v[int(onsets[i]*sampFreq/step):int(offsets[i]*sampFreq/step)]
    plt.plot(temp)
    
    localmin = argrelextrema( temp, np.less)
    localmin = localmin[0].tolist()
    lminval =[]
    for j in localmin:
        lminval.append(temp[j])
    plt.plot(localmin,lminval)
    
    left = int(len(lminval)/5)
    right = int((len(lminval)/5)*4)
    leftmin = (min(lminval[:left]))
    rightmin = (min(lminval[right:]))
    plt.plot(localmin[lminval.index(leftmin)],leftmin,'*')
    plt.plot(localmin[lminval.index(rightmin)],rightmin,'*')
    '''
    localmax = argrelextrema( temp, np.greater)
    localmax = localmax[0].tolist()
    #lminval =[]
    for j in localmax:
        lminval.append(temp[j])
    plt.stem(localmin+localmax,lminval)
    '''
    plt.show()
"""  