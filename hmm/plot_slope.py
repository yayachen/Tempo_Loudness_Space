# -*- coding: utf-8 -*-
"""
Created on Mon May  7 23:51:32 2018

@author: stanley
"""

import matplotlib.pyplot as plt
import numpy as np

vio_slope_2D = np.load("vio_slope_2D.npy")
vio_slope_list = list(np.load("vio_slope_list.npy"))
vib_slope_2D = np.load("vib_slope_2D.npy")
vib_slope_list = list(np.load("vib_slope_list.npy"))

all_slope = vio_slope_list+vib_slope_list

plt.boxplot(all_slope)
plt.show()

bins = 100

plt.hist(vio_slope_list, bins=bins, alpha=0.3, label='Normal')
plt.hist(vib_slope_list, bins=bins, alpha=0.3, label='Vibrato')
#plt.title('Random Gaussian data')
plt.xlabel('Slope')
plt.ylabel('count')
plt.legend(loc='upper right')
 
 
plt.show()
