# -*- coding: utf-8 -*-
"""
Created on Mon May  7 20:58:46 2018

@author: stanley
"""

import numpy as np
from sklearn.linear_model import LinearRegression
def LR(envData,n = 10):

    frame_size = int(len(envData)/n)
    x = np.array(range(frame_size))
    cnt = 0
    slope = []
    
    for i in range(10):
    
        
        y = np.array(envData[cnt:cnt+frame_size])
        #y = y*1.0/max(abs(y))
        
        lm = LinearRegression()
        lm.fit(np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1)))
        
        
        # 印出係數
        print(lm.coef_)
        
        # 印出截距
        print(lm.intercept_ )
        
        plt.plot(y)
        test = [ float(lm.coef_)*i + float(lm.intercept_)  for i in range(frame_size) ]
        plt.plot(test)
        plt.show()
        
        
        cnt = cnt+frame_size
        slope.append(float(lm.coef_))

    return slope

if __name__ == "__main__":

    
    import librosa
    import matplotlib.pyplot as plt

    r_dir = r"C:\Users\stanley\Desktop\SCREAM Lab\np&pd\rwc_violin_mono&vibrato"
    snd, sr = librosa.load(r_dir+"\\mono\\151\\vio151_03_3G_3A"+'.wav')
    #snd, sr = librosa.load(r_dir+"\\vibrato\\151\\vib151_03_3G_3A"+'.wav')
    waveData = snd*1.0/max(abs(snd))
    plt.plot(waveData)
    plt.show()
    import getEnvelope
    envData = getEnvelope.getEnvelope(waveData)
    plt.plot(envData)
    plt.show()
    
    slope = LR(envData,10)
    plt.plot(slope)