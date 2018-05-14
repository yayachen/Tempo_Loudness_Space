# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 03:13:25 2018

@author: stanley
"""
import os
import re

r_dir = r"C:\Users\stanley\Desktop\SCREAM Lab\np&pd\wave_w_o silence & new GT"
for root, sub, files in os.walk(r_dir):
    files = sorted(files)

    for f in files:       
        #w     = scipy.io.wavfile.read(os.path.join(root, f))
        dir = os.path.dirname(r_dir)
        #print(dir)
        #print(root)
        base=os.path.basename(f)
        print (root+"\\"+base)
        if "ground.txt" in base:
            song = base.split("_ground")[0]
            f2 = open(root+"\\"+base,'r+')
            fr2=f2.read()
            fr2 = re.split('\n|\t',fr2)
            fr2.pop()
            onsets = [float(fr2[i])  for i in range(len(fr2)) if i%5==0 ]
            offsets = [float(fr2[i])  for i in range(len(fr2)) if i%5==1]
            f2.close()
            
            fw = open(root+"\\"+song+"_feature.txt",'w+')
            for i in range(len(onsets)):
                #fw.write('note'+str(i+1)+"\t")
                fw.write(str(onsets[i])+"\t")
                fw.write(str(offsets[i])+"")
                fw.write("\n")
            fw.close()
        if not os.path.exists(dir):
           os.mkdir(base)        
           
        print('-------------------------')
