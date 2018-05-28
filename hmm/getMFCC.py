# -*- coding: utf-8 -*-
"""
Created on Mon May 14 22:20:00 2018

@author: stanley
"""
import librosa 
import numpy as np
#import scipy 

def getMFCC(wavpath):
    
    #sr, y = scipy.io.wavfile.read(wavpath)
    y, sr = librosa.load(wavpath)
    y = y*1.0/max(abs(y))
    mfccs  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    #mean = np.array([np.mean(mfccs[i]) for i in range(len(mfccs))])
    #proportion = [ mean[i]/mean[0] for i in range(1,len(mean))]
    
    n  = 10
    frame_size = len(mfccs[0])//10
    
    mfccs  = np.array([ np.mean(mfccs[j,i*frame_size : (i+1)*frame_size])  for j in range(len(mfccs)) for i in range(n)])
    
    return mfccs
    
if __name__== "__main__":
    
    import matplotlib.pyplot as plt

    r_dir = r"C:\Users\stanley\Desktop\SCREAM Lab\np&pd\rwc_violin_mono&vibrato"
    snd, sr = librosa.load(r_dir+"\\mono\\151\\vio151_03_3G_3A"+'.wav')
    #snd, sr = librosa.load(r_dir+"\\vibrato\\151\\vib151_61_5E_7C#"+'.wav')
    waveData = snd*1.0/max(abs(snd))
    plt.plot(waveData)
    plt.show()
    
    mfccs = getMFCC(snd,sr)
    plt.plot(mfccs)
    plt.show()
    
    import librosa.display
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    
