# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 04:01:19 2018

@author: stanley
"""

import os
import re
import numpy as np

f_all = []
r_dir = r"C:\Users\stanley\Desktop\SCREAM Lab\np&pd\wave_w_o silence & new GT"
for root, sub, files in os.walk(r_dir):
    files = sorted(files)
    
    for f in files:       
        #w     = scipy.io.wavfile.read(os.path.join(root, f))
        dir = os.path.dirname(r_dir)
        base=os.path.basename(f)
        #print (root+"\\"+base)
        if "_feature.txt" in base:
            song = base.split("_feature.txt")[0]
            f = open(root+"\\"+base,'r+')
            fr=f.read()
            fr = re.split('\n|\t',fr)
            fr.pop()
            fr = [float(fr[i]) for i in range(len(fr))]
            f.close()
            f_all += fr
            
        if not os.path.exists(dir):
           os.mkdir(base)        
           
        #print('-------------------------')
f_all = np.array(f_all).reshape(len(f_all)//2,2)
  



import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

#%% generating samples
k=5 # number of clusters
n_samples = 1500  
random_state = 17

X, y = make_blobs(n_samples=n_samples, centers=k, n_features=7,random_state=random_state)
X = f_all
#%% call kmeans api
kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=0 )
# Fitting with inputs
kmeans.fit(X)
# Predicting the clusters
labels = kmeans.predict(X)
# Getting the cluster centers
C_f = kmeans.cluster_centers_


colors = ['y', 'g', 'b', 'c', 'r', 'm', 'orange', 'cyan', 'indigo', 'teal']
fig = plt.figure()
for m in range(k):
        datap = np.array([X[j] for j in range(len(X)) if labels[j] == m])
        if (np.isnan(np.mean(datap))):
            plt.scatter(C_f[m, 0], C_f[m, 1], marker='*', s=200, c=colors[m])
        else: 
            plt.scatter(datap[:, 0], datap[:, 1], s=1, c=colors[m])
        plt.scatter(C_f[m, 0], C_f[m, 1], marker='*', s=200, c=colors[m])
plt.savefig("kmeans_test.png",dpi=500)