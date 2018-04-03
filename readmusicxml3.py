# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 13:45:11 2018

@author: stanley
"""
def readmusicxml(midipath = 'C:/Users/stanley/Desktop/SCREAM Lab/np&pd/10violin/midi/02_violin.xml'):
    import xml.etree.cElementTree as ET
    #tree = ET.ElementTree(file='C:/Users/stanley/Desktop/SCREAM Lab/np&pd/10violin/midi/02_violin.xml')
    tree = ET.ElementTree(file=midipath)
    note_value = {'quarter':1,
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
                notes_duration[-1] +=note_value[elem.text]
                tieds[-1].append(note_value[elem.text])
            elif is_rest:
                notes_duration[-1] +=note_value[elem.text]
                is_rest = False
            else: 
                # print(len(notes_duration))
                notes_duration.append(note_value[elem.text])
                #if elem.tag =='beats':
                #print(elem.attrib,elem.text)
    
    Bar.append(note_cnt)
    del(Bar[0])
    #e = tree.findall('part/measure')