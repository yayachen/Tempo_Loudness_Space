# -*- coding: utf-8 -*-
"""
Created on Mon May 14 21:56:41 2018

@author: stanley
"""






import numpy as np
from hmmlearn import hmm

vio_mfcc_2D = np.load("vio_mfcc_2D.npy")
vib_mfcc_2D = np.load("vib_mfcc_2D.npy")

##3D
train = all_mfcc = np.row_stack((vio_mfcc_2D,vib_mfcc_2D)) 
##2D
if len(all_mfcc.shape) == 3:
    nsamples, nx, ny = all_mfcc.shape
    train = all_mfcc = all_mfcc.reshape((nsamples,nx*ny))

## split train test
from sklearn.model_selection import train_test_split
train ,test = train_test_split(all_mfcc, test_size=0.01, random_state=42)



states = ["normal", "vibrato"]
n_states = len(states)


## Gaussian
modelvio = hmm.GMMHMM(n_components= 10, n_iter=1000 , covariance_type="diag")#spherical")
#X2 = train 


modelvio.fit(vio_mfcc_2D)
hidden_states = modelvio.predict(vio_mfcc_2D)

print (modelvio.score(vio_mfcc_2D))

print (modelvio.startprob_)
print (modelvio.transmat_)
#print (modelvio.means_)
#print (modelvio.covars_)
print(hidden_states)

## vib

modelvib = hmm.GMMHMM(n_components= 13, n_iter=1000 , covariance_type="diag")#spherical")
#X2 = train 


modelvib.fit(vib_mfcc_2D)
hidden_states = modelvib.predict(vib_mfcc_2D)

print (modelvib.score(vib_mfcc_2D))

print (modelvib.startprob_)
print (modelvib.transmat_)
#print (modelvib.means_)
#print (modelvib.covars_)
print(hidden_states)

'''
seen = np.array([all_slope[2]])
logprob, playingstyle = model2.decode(seen, algorithm="viterbi")
print([ states[p] for p in playingstyle])

seen = np.array([all_slope[361]])
logprob, playingstyle = model2.decode(seen, algorithm="viterbi")
print([ states[p] for p in playingstyle])
'''
print ("==================================")

vioout = []
for i in range(len(vio_mfcc_2D)):
    if modelvio.score(vio_mfcc_2D[i:i+1]) > modelvib.score(vio_mfcc_2D[i:i+1]):
        vioout.append(0)
    else:
        vioout.append(1)

vibout = []
for i in range(len(vio_mfcc_2D)):
    if modelvib.score(vib_mfcc_2D[i:i+1]) > modelvio.score(vib_mfcc_2D[i:i+1]):
        vibout.append(1)
    else:
        vibout.append(0)
'''            
import networkx as nx
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt

def _get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx,col)] = Q.loc[idx,col]
    return edges

hidden_states = states
pi = list(model2.startprob_)

a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
a_df.loc[hidden_states] = model2.transmat_

#print(a_df)
a = a_df.values
#print('\n', a, a.shape, '\n')
#print(a_df.sum(axis=1))


observable_states = observations

b_df = pd.DataFrame(columns=observable_states, index=hidden_states)
b_df.loc[hidden_states] = model2.emissionprob_

#print(b_df)

b = b_df.values
#print('\n', b, b.shape, '\n')
#print(b_df.sum(axis=1))


# create graph edges and weights

hide_edges_wts = _get_markov_edges(a_df)
pprint(hide_edges_wts)

emit_edges_wts = _get_markov_edges(b_df)
pprint(emit_edges_wts)







# create graph object
G = nx.MultiDiGraph()

# nodes correspond to states
G.add_nodes_from(hidden_states)
print(f'Nodes:\n{G.nodes()}\n')

# edges represent hidden probabilities
for k, v in hide_edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    v = round(v,5)
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)

# edges represent emission probabilities
for k, v in emit_edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    v = round(v,5)
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
    
#print(f'Edges:')
#pprint(G.edges(data=True))    

pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='neato')
nx.draw_networkx(G, pos)

# create edge labels for jupyter plot but is not necessary
emit_edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G , pos, edge_labels=emit_edge_labels,font_size = 6, width = 0.5)
nx.drawing.nx_pydot.write_dot(G, 'pet_dog_hidden_markov.dot')
plt.savefig("hmm.png",dpi = 1500) 
plt.show()   
'''