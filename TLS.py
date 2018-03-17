# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 11:06:49 2018

@author: stanley
"""


import pylab as pl
from matplotlib import animation
import math
import numpy as np
from scipy.io import wavfile
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
        #try:
           #volume[i] = 10*np.log10(np.sum(curFrame*curFrame))
        #except:
            #volume[i] = 10*np.log10(np.sum(curFrame*curFrame)+0.00000001)
    return volume
    #return np.mean(volume)
"""    
def replaceZeroes(data):
  min_nonzero = np.min(data[np.nonzero(data)])
  data[data == 0] = min_nonzero
  return data
"""

####讀取wavedata(正規化)
wavepath = '../10violin/wave/' 
midipath = '../10violin/midi/' 
outputpath = 'TLS_result/'
song = '01-2'
sampFreq, snd = wavfile.read(wavepath+song+'.wav')
waveData = snd*1.0/max(abs(snd))

frameSize = 256
overLap = 128
smooth_size = 1

##讀取ground truth與小節音符個數
import xml.etree.cElementTree as ET
tree = ET.ElementTree(file=midipath+'01_violin.xml')
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



#Barlen = [] 
BarCnt = [0] 
Bar_onset = []
#BarSum = 0


for i in Bar:
     #Barlen.append(int((offsets[BarCnt+i-1]-onsets[BarCnt])*sampFreq))   
     #BarSum = BarSum + Barlen[-1]
     #Bar_onset.append(BarSum)
     #print(BarCnt[-1])
     Bar_onset.append(onsets[BarCnt[-1]])
     BarCnt.append(BarCnt[-1]+i)  
     
## 找出延長線有跨小節(扣除最後一小節05)     
for i in range(len(BarCnt[:-1])):
    print(i)
    for j in tieds:
         if BarCnt[i]-1 == j[0]:
             #print("tied")
             #print(BarCnt[i]-1)
             #print(onsets[BarCnt[i]],onsets[BarCnt[i]-1])
             #print(onsets[BarCnt[i]]-onsets[BarCnt[i]-1])
             #print((onsets[BarCnt[i]]-onsets[BarCnt[i]-1])*(j[2]/(j[1]+j[2])))
             #print(Bar_onset[i])
             Bar_onset[i]=Bar_onset[i]-(onsets[BarCnt[i]]-onsets[BarCnt[i]-1])*(j[2]/(j[1]+j[2]))
             #print(Bar_onset[i])
             
onsetslen = []
for i in range(len(onsets)):
    try:
        #onsetslen.append(int((offsets[i]-onsets[i])*sampFreq))
        onsetslen.append(int((onsets[i+1]-onsets[i])*sampFreq))
    except IndexError:
        onsetslen.append(int((offsets[i]-onsets[i])*sampFreq))


###08
#time_signature = 3
track_level = min(time_value) #eighth note
        
###03
#time_signature = 4
#track_level = 0.5 #eighth note  

Bar_mean_len = len(waveData)/sampFreq/len(Bar)
note_mean_len = len(waveData)/sampFreq/sum(time_value)



tl_len = []
tl_len_cnt = []
cnt = 0
for i in range(len(time_value)):
    tl_time =  (onsetslen[i]/sampFreq)/(time_value[i]/track_level)
    #print(tl_time)
    for j in range(int(time_value[i]/track_level)):
        #tl_len.append(tl_len_cnt)
        tl_len.append(tl_time)
        tl_len_cnt.append(cnt)
        cnt=cnt + tl_time
        

v = [] 
bpm = []
for i in range(len(tl_len)):
    onset = int(tl_len_cnt[i]*sampFreq)
    offset = int((tl_len_cnt[i]+tl_len[i])*sampFreq)
    #print((offset-onset)/sampFreq)
    db = np.mean(calVolumeDB(waveData[onset:offset],frameSize,overLap))
    #v.append(math.pow(2,((db-40)/10)))
    v.append(db)
    bpm.append(60/tl_len[i]/(1/track_level))
  


import pylab as plt
import scipy.ndimage.filters as filters
time = np.arange(0,len(waveData)/sampFreq,0.001)

nv = np.interp(time,tl_len_cnt,v)
nbpm = np.interp(time,tl_len_cnt,bpm)
sigma = note_mean_len*1000*smooth_size#/2        ##window_size
fv = filters.gaussian_filter1d(nv,sigma=sigma)
fbpm = filters.gaussian_filter1d(nbpm,sigma=sigma)

#####onset###
t_onsets = [int(tl_len_cnt[i]*1000)  for i in range(len(tl_len_cnt))]
t_note_v= [nv[int(tl_len_cnt[i]*1000)]  for i in range(len(tl_len_cnt))]    
t_note_bpm = [nbpm[int(tl_len_cnt[i]*1000)] for i in range(len(tl_len_cnt))]

#####Bar###
t_bar_v= [fv[int(Bar_onset[i]*1000)]  for i in range(len(Bar_onset))]    
t_bar_bpm = [fbpm[int(Bar_onset[i]*1000)] for i in range(len(Bar_onset))]
    

plt.figure()

plt.subplot(211)
for i in Bar_onset:
        plt.plot([i, i], [max(v), min(v)],'--b')
plt.scatter(tl_len_cnt,t_note_v,s=10,label='onset')
plt.plot(time,nv,label='loudness data')
plt.plot(time,fv,label='smoothed : 1/4 Bar',linewidth=2)
plt.ylabel("Loudness (dB)")
plt.xlabel("time (s))")
#plt.legend(loc='best')

plt.subplot(212)
for i in Bar_onset:
        plt.plot([i, i], [max(bpm), min(bpm)],'--b')
plt.scatter(tl_len_cnt,t_note_bpm,s=10,label='onset')
plt.plot(time,nbpm,label='tempo data')
plt.plot(time,fbpm,label='smoothed : 1/4 Bar',linewidth=2)
plt.ylabel("Tempo (bpm)")
plt.xlabel("time (s)")
#plt.legend(loc='best')

plt.savefig(outputpath+song+'T&L.png',dpi=300)
plt.show()





#plt.subplot(211)    
plt.plot(fbpm, fv,color='orange',lw=2,zorder=2)

T = [i for i in range(len(Bar_onset))]  
plt.scatter(t_bar_bpm,t_bar_v,alpha=0.5,s=50,c =T,cmap=plt.cm.Oranges,zorder=2,label='Bar');
plt.ylabel("Loudness (dB)")
plt.xlabel("Tempo (bpm)")
plt.colorbar()
for i ,num in enumerate(range(len(Bar_onset))):
    plt.annotate(num,(t_bar_bpm[i],t_bar_v[i]))   
plt.legend()
plt.savefig(outputpath+song+'TLS.png',dpi=300)
plt.show()


"""

frames = int(len(fv)/100)
sbpm =[]
sv= []
for i in range(frames):
    if i <= 10:
        sbpm.append( fbpm[i*100])
        sv.append(fv[i*100])
    else:
        sbpm.append(fbpm[(i-10)*100])
        sv.append(fv[(i-10)*100])

plt.scatter(sbpm,sv,alpha=0.5,s=50,zorder=2)
plt.scatter(t_bar_bpm,t_bar_v,alpha=0.9,s=250,c =T,cmap=pl.cm.Oranges,zorder=2,label='Bar');
plt.ylabel("Loudness (dB)")
plt.xlabel("Tempo (bpm)")
plt.colorbar()
for i ,num in enumerate(range(len(Bar_onset))):
    pl.annotate(num,(t_bar_bpm[i],t_bar_v[i]))   
plt.legend()
plt.show()

"""


fig = pl.figure()
ax1 = fig.add_subplot(1,1,1)
ax2 = fig.add_subplot(1,1,1)




ax = fig.add_subplot(1,1,1)
frames = int(len(fv)/100)
TLS = np.zeros(frames, dtype =[('position', float , 2),
                                ('size',     float, 1),
                              ('reduce',   float, 1),
                              ('color',    float, 4),
                              ('annotation' ,  str,3)])
        
#rain_drops['position'] = np.random.uniform(0, 1, (n_drops, 2))
#rain_drops['growth'] = np.random.uniform(50, 200, n_drops)

TLS['position'][:, 0] = [ fbpm[i*100] for i in range(len(TLS))]
TLS['position'][:, 1] = [ fv[i*100] for i in range(len(TLS))]
TLS['annotation'] = [str(i+1) for i in range(len(TLS))]



time_template = 'time = %.1fs'
time_text = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)


scat = ax.scatter(TLS['position'][:, 0], TLS['position'][:, 1],
                  s=TLS['size'], lw=1, edgecolors=TLS['color'],
                  facecolors='coral')

annotation = ax.annotate('',#TLS['annotation'][0], 
                         xy=(1,0)#, xytext=(-1,0),arrowprops = {'arrowstyle': ""}
)


pl.ylabel("Loudness(dB)")
pl.xlabel("Tempo(bpm))")

    
    
#cur, = ax2.plot(fbpm, fv,color='orange',lw=2,zorder=2)
#sca, = ax1.plot(t_bar_bpm,t_bar_v,'o',alpha=0.8,zorder=3)

Bar_onset_s = [int(v*10) for v in Bar_onset]


Bar_mean = int(note_mean_len*10)
min_size = 3
reduce_size = 10
cur_size = (Bar_mean*reduce_size)+min_size
reduce_size = 10
reduce_color = 1/(Bar_mean+0.01)
Bar_onset_s.append(len(TLS))


for i in range(len(TLS)):
    for j in range(len(Bar_onset_s)-1):
        if i >= Bar_onset_s[j] and i < Bar_onset_s[j+1]:
            TLS['annotation'][i] = str(j+1)
            
            
def animate(i):
    current_index = i 
    TLS['color'][current_index] = (1, 0.1, 0.1, 1)
    TLS['size'][current_index] = cur_size
    
    if i >= Bar_mean and i < 2*Bar_mean:
        TLS['size'][0:i-Bar_mean] -= reduce_size
        TLS['color'][0:i-Bar_mean,0] -= reduce_color
    if i >= 2*Bar_mean :
        TLS['size'][i-2*Bar_mean:i-Bar_mean] -= reduce_size
        TLS['color'][i-2*Bar_mean:i-Bar_mean,0] -= reduce_color
    for j in Bar_onset_s:
        if i >= j:
            TLS['size'][j] =   cur_size*1.5
    
    
    
            
    '''
    if i >= Bar_mean:
        TLS['size'][0:i-Bar_mean] -= reduce_size
        for j in range(i-Bar_mean):
            if TLS['size'][j] < min_size:
                TLS['size'][j] = min_size
    '''
    
    scat.set_edgecolors(TLS['color'])
    scat.set_sizes(TLS['size'])
    scat.set_offsets(TLS['position'])  
    scat.set
    
    annotation.set_position(TLS['position'][i]) 
    annotation.xy=TLS['position'][i]
    annotation.set_text(TLS['annotation'][i])
    
    time_text.set_text(time_template % (i/10))
    
    return scat,annotation,

# Init only required for blitting to give a clean slate.
def init():
    
    return scat,


ani = animation.FuncAnimation(fig=fig, func=animate, frames=frames, init_func=init,
                              interval=100, blit=False)

ani.save(outputpath+song+'.mp4' ,writer='ffmpeg_file',dpi=200)




from moviepy.editor import VideoFileClip ,AudioFileClip
 
clip1 = VideoFileClip(outputpath+song+'.mp4')

clip2 = AudioFileClip(wavepath+song+'.wav')
 

new_video = clip1.set_audio(clip2)

new_video.write_videofile(outputpath+song+'Tempo_Loudness_Space.mp4')
