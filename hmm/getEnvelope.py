# -*- coding: utf-8 -*-
"""
Created on Mon May  7 18:07:22 2018

@author: stanley
"""

from scipy.signal import butter, lfilter , filtfilt
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
        
    b,a = butter(3,150/(44100/2),'low')
    outputSignal = filtfilt(b,a,outputSignal)
    
    return outputSignal