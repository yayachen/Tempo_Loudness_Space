# -*- coding: utf-8 -*-
"""
Created on Tue May  8 01:18:09 2018

@author: stanley
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 01:27:30 2018

@author: stanley
"""

def equal_width_into4(x):
    #level_dict = {0:'significantly decrease',1:'slightly decrease',2:'slightly increase',3:"significantly increase"}
    w = (max(x)-min(x))/4
    temp = []
    for i in range(len(x)):
        cnt = 4
        for n in range(1,5):
            if x[i] <= min(x)+(n*w):
                cnt -= 1
        #temp.append(level_dict[cnt]) ## LOW,Medium,High
        temp.append(cnt) ##0,1,2
    return temp , [min(x),min(x)+w,min(x)+2*w,min(x)+3*w]

def test(x):
    temp = []
    for i in x:
        if i < -0.00005:
            temp.append(0)
        elif i < 0:
            temp.append(1)
        elif i < 0.00005:
            temp.append(2)
        else:
            temp.append(3)
    return temp



import numpy as np
from hmmlearn import hmm

vio_slope_2D = np.load("vio_slope_2D.npy")
vio_slope_list = list(np.load("vio_slope_list.npy"))
vib_slope_2D = np.load("vib_slope_2D.npy")
vib_slope_list = list(np.load("vib_slope_list.npy"))

all_slope = vio_slope_list+vib_slope_list
all_slope,b = equal_width_into4(all_slope)
#all_slope = test(all_slope)
all_slope = np.array(all_slope)
n = 10
all_slope = all_slope.reshape([len(all_slope)//n,n])



states = ["normal", "vibrato"]
n_states = len(states)

observations = ["significantly decrease","slightly decrease","slightly increase","significantly increase"]
n_observations = len(observations)
model2 = hmm.MultinomialHMM(n_components=n_states, n_iter=20, tol=0.01)
X2 = all_slope #np.array([[0,1,2,1],[3,0,0,1],[1,0,1,1]])

"""
model2.fit(X2)
print (model2.startprob_)
print (model2.transmat_)
print (model2.emissionprob_)
print (model2.score(X2))
"""

max_score = -10000
for i in range(10):
    model2.fit(X2)
    print (model2.score(X2))
    if model2.score(X2) > max_score:
        max_startprob_ = model2.startprob_
        max_transmat_ = model2.transmat_
        max_emissionprob_ = model2.emissionprob_
        max_score = model2.score(X2)

model2.startprob_ = max_startprob_
model2.transmat_ = max_transmat_        
model2.emissionprob_ = max_emissionprob_  
#model2.score(X2) = max_score

print (model2.startprob_)
print (model2.transmat_)
print (model2.emissionprob_)
print (model2.score(X2))

seen = np.array([all_slope[2]]).T
logprob, playingstyle = model2.decode(seen, algorithm="viterbi")
print([ states[p] for p in playingstyle])

seen = np.array([all_slope[362]]).T
logprob, playingstyle = model2.decode(seen, algorithm="viterbi")
print([ states[p] for p in playingstyle])

print ("==================================")


            
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

