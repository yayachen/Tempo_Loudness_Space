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

r_dir = r"C:\Users\stanley\Desktop\SCREAM Lab\np&pd\rwc_violin_mono&vibrato"

vio_slope_2D = []
vio_slope_list = []

vib_slope_2D = []
vib_slope_list = []

for root, sub, files in os.walk(r_dir):
    files = sorted(files)

    for f in files:       
        #w     = scipy.io.wavfile.read(os.path.join(root, f))
        dir = os.path.dirname(r_dir)
        base=os.path.basename(f)
        print (root+"\\"+base)
        
        y, sr = librosa.load(root+"\\"+base)
        waveData = y*1.0/max(abs(y))
        envData = getEnvelope(waveData)
        
        note_slope = LR(envData)
        if (base[0:3] == 'vio'):
            vio_slope_2D.append(note_slope)
            vio_slope_list += note_slope
            
        if (base[0:3] == 'vib'):
            vib_slope_2D.append(note_slope)
            vib_slope_list += note_slope
        #if not os.path.exists(dir):
           #os.mkdir(base)        
           
        print('-------------------------')
        
np.save("vio_slope_2D.npy",vio_slope_2D)
np.save("vio_slope_list.npy",vio_slope_list)
np.save("vib_slope_2D.npy",vio_slope_2D)
np.save("vib_slope_list.npy",vio_slope_list)



"""
import numpy as np
from hmmlearn import hmm

states = ["box 1", "box 2","box 3"]
n_states = len(states)

observations = ["red", "white","blue"]
n_observations = len(observations)
model2 = hmm.MultinomialHMM(n_components=n_states, n_iter=20, tol=0.01)
X2 = np.array([[1],[0]])
from sklearn.preprocessing import LabelEncoder
#X2 = LabelEncoder().fit_transform(X2)
model2.fit(X2)
print (model2.startprob_)
print (model2.transmat_)
print (model2.emissionprob_)
print (model2.score(X2))

model2.fit(X2)
print (model2.startprob_)
print (model2.transmat_)
print (model2.emissionprob_)
print (model2.score(X2))
model2.fit(X2)
print (model2.startprob_)
print (model2.transmat_)
print (model2.emissionprob_)
print (model2.score(X2))
"""