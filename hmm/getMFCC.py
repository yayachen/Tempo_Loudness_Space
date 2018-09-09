# -*- coding: utf-8 -*-
"""
Created on Mon May 14 22:20:00 2018

@author: stanley
"""
import librosa 
import numpy as np
from scipy import spatial

def getMFCC(wavpath,n_mfcc=13,n = 10):
    
    #sr, y = scipy.io.wavfile.read(wavpath)
    y, sr = librosa.load(wavpath,sr = 44100)
    y = y*1.0/max(abs(y))
    mfccs  = librosa.feature.mfcc(y=y,n_mfcc=n_mfcc, sr=sr, dct_type=2)
    #mean = np.array([np.mean(mfccs[i]) for i in range(len(mfccs))])
    #proportion = [ mean[i]/mean[0] for i in range(1,len(mean))]
    
    frame_size = len(mfccs[0])//n
    #normalization
    #mfccs /= np.max(np.abs(mfccs),axis=0)
    #mfccs = (mfccs+1)*1000
    #mean
    mfccs  = np.array([ np.mean(mfccs[j,i*frame_size : (i+1)*frame_size])  for j in range(len(mfccs)) for i in range(n)])
    
    ##cos
    #mfccs = np.array([ 1 - spatial.distance.cosine(mfccs[:,i-1], mfccs[:,i]) for i in range(1,len(mfccs[0]))])
    #mfccs = np.array([ np.mean(mfccs[i*frame_size:(i+1)*frame_size])  for i in range(n)])
    return mfccs
    
if __name__== "__main__":
    
    import matplotlib.pyplot as plt

    r_dir = r"C:\Users\stanley\Desktop\SCREAM Lab\np&pd\rwc_violin_mono&vibrato"
    wavpath = "../../RWC\\151\\151VNNOF\\151VNNOF_01_3G_3G.wav"
    waveData, sr = librosa.load(wavpath,sr = 44100)
    plt.plot(waveData)
    plt.show()
    
    mfccs = getMFCC(wavpath)
    plt.plot(mfccs)
    plt.show()
    
    """
    import librosa.display
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    """
