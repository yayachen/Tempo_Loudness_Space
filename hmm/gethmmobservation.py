# -*- coding: utf-8 -*-
"""
Created on Mon May  7 18:09:29 2018

@author: stanley
"""

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from getEnvelope import getEnvelope
from LinearRegressiontest import LR
from getMFCC import getMFCC

r_dir = r"C:\Users\stanley\Desktop\SCREAM Lab\np&pd\rwc_violin_mono&vibrato"

vio_mfcc_2D = []
vib_mfcc_2D = []


for root, sub, files in os.walk(r_dir):
    files = sorted(files)

    for f in files:       
        #w     = scipy.io.wavfile.read(os.path.join(root, f))
        dir = os.path.dirname(r_dir)
        base=os.path.basename(f)
        print (root+"\\"+base)
        
        y, sr = librosa.load(root+"\\"+base)
        waveData = y*1.0/max(abs(y))
        note_mfcc = getMFCC(y,sr)
        
        if (base[0:3] == 'vio'):
            vio_mfcc_2D.append(note_mfcc)
            #vio_mfcc_list += note_mfcc
            
        if (base[0:3] == 'vib'):
            vib_mfcc_2D.append(note_mfcc)
            #vib_mfcc_list += note_mfcc
        
        print('-------------------------')
        
np.save("vio_mfcc_2D.npy",vio_mfcc_2D)
np.save("vib_mfcc_2D.npy",vib_mfcc_2D)




