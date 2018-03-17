# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 12:27:19 2017

@author: stanley
"""

#!/usr/bin/env python
"""
Open a MIDI file and print every message in every track.
Support for MIDI files is still experimental.
"""


import midi 
import matplotlib.pyplot as plt
song = midi.read_midifile("test.mid")
#song = midi.read_midifile("C:/Users/stanley/Desktop/SCREAM Lab/np&pd/Tempo_Loudness_Space/midi/05_violin.mid")
#print(song)
song.make_ticks_abs()
#print('###################')
#print(song)

tracks = []
notes=[]
bpm = 0
for track in song:
    setTempo = [Event for Event in track if Event.name == 'Set Tempo' ]
    print(setTempo[0].bpm)
    bpm = setTempo[0].bpm

ppq = song.resolution
b2s = (600000/(bpm*ppq))/10000
'''
for track in song:
    notes = [note for note in track if note.name == 'Note On']
    pitch = [note.pitch for note in notes]
    tick = [note.tick for note in notes]
    ppq = song.resolution
    second = [ b2s*tk for tk in tick] 
    tracks += [tick, pitch]
plt.plot(*tracks)
plt.show()
print(song.resolution)
'''
cnt = 0
for track in song:
    for note in track:
        if note.name == 'Note On' and note.data[1] == 80:
            cnt=cnt+1




import numpy as np
midiData = np.zeros(cnt, dtype=[('start beats',    float,1),
                              ('duration beats',   float, 1),
                              ('beats',            float, 1),
                              ('channel',          int, 1),
                              ('pitch',            int, 1),
                              ('velocity' ,        int,1),
                              ('start seconds',    float,1 ),
                              ('duration seconds', float,1),
                              ('bar',              int ,1)])

notes = []
note_temp = 0
for track in song:
    notes = [note for note in track if note.name == 'Note On']
    pitch = [note.pitch for note in notes]
    tick = [note.tick for note in notes]
    ppq = song.resolution
    second = [ b2s*tk for tk in tick] 
    tracks += [tick, pitch, second]   
    
    
for i in range(len(tracks[0])):
    if ((i%2) == 1):
        index = int(i/2)
        midiData['start beats'][index] = tracks[0][i-1]/ppq
        midiData['duration beats'][index] = (tracks[0][i]-tracks[0][i-1])/ppq
        midiData['channel'][index] = notes[i].channel +1
        midiData['pitch'][index] = notes[i].pitch
        midiData['velocity'][index] = notes[i-1].velocity 
        midiData['start seconds'][index] = tracks[2][i-1] 
        midiData['duration seconds'][index] = (tracks[2][i]-tracks[2][i-1])
        temp_min = 10
        temp_beat = 0
        for i in range(10):  #長拳音符(0)-128分音符(9)
            temp = abs(midiData['duration beats'][index]-pow(2,(4-i)))
            if temp < temp_min:
                temp_min = temp
                temp_beat = i
        midiData['beats'][index] = pow(2,(4-temp_beat))
        
events = [event for track in song for event in track if event.name =='Note On' or (event.name == 'Time Signature')]

cnt = 0
beat_cnt = 0   
beat_acc = 0 
bar_cnt = 0
Bar = []
for event in events:
    if event.name == 'Time Signature':
        #print(event.data[0])
        #print(4/pow(2,event.data[1]))
        meter_len = event.data[0]*(4/pow(2,event.data[1]))
        print('meter='+str(meter_len))
    if  (event.name =='Note On' and event.velocity != 0):
        if midiData['start beats'][cnt] < (beat_acc+meter_len ):
            beat_cnt+=1
        else:
            Bar.append(beat_cnt)
            beat_cnt = 0
            beat_acc += meter_len
        cnt+=1
    print(beat_acc)
    '''
    if (event.name =='Note On' and event.velocity != 0):
        print(cnt)
        beat_acc += midiData['beats'][cnt]
        beat_cnt += 1
        cnt+= 1
        if beat_acc == meter_len:
            print('*****')
            Bar.append(beat_cnt)
            beat_acc = 0
            beat_cnt = 0
    '''    
#print(song.)
'''
from mido import MidiFile

import MIDI_
MIDI_. midi2opus("test.mid")
filename = "test.mid"#"01-2.mid"
midi_file = MidiFile(filename)

for i, track in enumerate(midi_file.tracks):
    print('=== Track {}\n'.format(i))
    for message in track:
        print('  {!r}\n'.format(message))
        
        
        
import mingus
from mingus.midi import MidiFileIn
MidiFileIn.read_NoteContainer("test.mid")
#mingus.midi.midi_file_in.MIDI_to_Composition(test.mid)
'''
'''
for i, track in enumerate(midi_file.tracks):
    print('Track {}: {}'.format(i, track.name))
    for msg in track:
        print(msg)
'''
'''
f = open('YA/noteBar_TEST.txt','r')
fr = f.read()
Bar = fr.split('\n')
Bar.pop()
Bar[:]=[int(i)  for i in Bar]

import re

onsetpath = '01-2_ground.txt'
f2 = open(onsetpath,'r')
fr2=f2.read()

fr2 = re.split('\n|\t',fr2)

fr2.pop()


if('ground' in onsetpath):
     onsets = [float(fr2[i])  for i in range(len(fr2)) if i%5==0 ]
     offsets = [float(fr2[i])  for i in range(len(fr2)) if i%5==1]
else:
     onsets = [float(fr2[i])  for i in range(len(fr2)) if i%6==2 ]
     offsets = [float(fr2[i])  for i in range(len(fr2)) if i%6==3]


print(onsets)
print(offsets)

Barlen = [] 
BarCnt = 0 
for i in Bar:
    if((BarCnt+i) < len(onsets)):
        Barlen.append(onsets[BarCnt+i]-onsets[BarCnt])
    else:
        Barlen.append(offsets[BarCnt+i-1]-onsets[BarCnt])  
    BarCnt+=i
print(Barlen)
'''