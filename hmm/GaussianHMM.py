# -*- coding: utf-8 -*-
"""
Created on Mon May 14 21:56:41 2018

@author: stanley
"""


import numpy as np
from hmmlearn import hmm
from sklearn.model_selection import KFold 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz

all_playing_style = ['VNNO','VNNV','VNSP','VNTA','VNPC','VNSO']
#all_playing_style = ['VNNO','VNNV']


HMM_dict = {}
D = 13
n = 10
for p in all_playing_style:
    vars()[p] = np.load(p+str(D)+'D'+str(n)+"n"+'.npy')
    #vars()[p] = np.load(p+'_cos.npy')*1000
    
###
VNNOV = np.concatenate((VNNO,VNNV))
all_playing_style.pop(0)
all_playing_style[0] ='VNNOV' 
###    

"""
##3D
train = all_mfcc = np.row_stack((vio_mfcc,vib_mfcc)) 
##2D
if len(all_mfcc.shape) == 3:
    nsamples, nx, ny = all_mfcc.shape
    train = all_mfcc = all_mfcc.reshape((nsamples,nx*ny))
"""


states = ["normal", "vibrato"]
n_states = len(states)

n_ALL_accuracy = []

start,end = 12,13
N = 5 #iterater for k
for j in range(start,end):
    
    ## Gaussian HMM model
    ##### vio 
    #modelvio = hmm.GMMHMM(n_components= j, n_iter=1000 , covariance_type="diag")#spherical")
    ##### vib
    #modelvib = hmm.GMMHMM(n_components= j, n_iter=1000 , covariance_type="diag")#spherical")
    for p in all_playing_style:
        HMM_dict[p] = hmm.GMMHMM(n_components= j, n_iter=500 , covariance_type="diag")
    
    for n in range(N):
        testdata = []
        traindata = []
        conf = []
        accuracy = []
        
        
        #KFold  
        k = 5 
        kf = KFold(n_splits= k,shuffle=True) 
        
        train_dict = {}
        test_dict = {}
        
        for p in all_playing_style:
            train_dict[p] = []
            test_dict[p] = []
            for train_index , test_index in kf.split(vars()[p]):  
                train_dict[p].append(train_index)
                test_dict[p].append(test_index)

        
        for i in range(k):
            
            
            ##### train 
            #modelvio.fit(vio_mfcc[kf_vio_train[i]])
            #modelvib.fit(vib_mfcc[kf_vib_train[i]])
            testtemp = []
            traintemp = []
            for p in all_playing_style:
                HMM_dict[p].fit(vars()[p][train_dict[p][i]])
                #print(p,HMM_dict[p].score(vars()[p][train_dict[p][i]]))
                testtemp.append(vars()[p][test_dict[p][i]])
                traintemp.append(vars()[p][train_dict[p][i]])
                
            #### merge testdata
            #testdata.append(np.concatenate((vio_mfcc[kf_vio_test[i]],vib_mfcc[kf_vib_test[i]]),axis=0))
            
            """
            #tree
            traindata.append(np.concatenate(traintemp,axis=0))
            ground_truth = sum([ [all_playing_style.index(p)]*len(train_dict[p][i]) 
                                for p in all_playing_style],[])  ##2d to 1d
            clf = tree.DecisionTreeClassifier()

            out = []
            for v in traindata[i]:
                scores = []
                for p in all_playing_style:
                    while True:
                        try:        
                            scores.append(HMM_dict[p].score([v]))
                            break
                        except ValueError:
                            HMM_dict[p].fit(vars()[p][train_dict[p][i]])
                #print(scores)
                out.append(scores)
            clf = clf.fit(out,ground_truth)
            dot_data = tree.export_graphviz(clf, out_file=None) 
            graph = graphviz.Source(dot_data) 
            graph.render("tree") 
            dot_data = tree.export_graphviz(clf, out_file=None, 
                         filled=True, rounded=True,  
                         special_characters=True)  
            graph = graphviz.Source(dot_data) 
            #tree
            
            #tree test
            testdata.append(np.concatenate(testtemp,axis=0))
            
            out = []
            for v in testdata[i]:
                scores = []
                for p in all_playing_style:
                    while True:
                        try:        
                            scores.append(HMM_dict[p].score([v]))
                            break
                        except ValueError:
                            HMM_dict[p].fit(vars()[p][train_dict[p][i]])
                #print(scores)
                out.append(scores)
            out = clf.predict(out)
            #tree test
            """
            
            #max
            
            testdata.append(np.concatenate(testtemp,axis=0))
            
            out = []
            for v in testdata[i]:
                scores = []
                for p in all_playing_style:
                    while True:
                        try:        
                            scores.append(HMM_dict[p].score([v]))
                            break
                        except ValueError:
                            HMM_dict[p].fit(vars()[p][train_dict[p][i]])
                #print(scores)
                out.append(scores.index(max(scores)))
             
            
            #confusion_matrix
            ground_truth = sum([ [all_playing_style.index(p)]*len(test_dict[p][i]) 
                                for p in all_playing_style],[])  ##2d to 1d
            #ground_truth = [0]*len(kf_vio_test[i])+[1]*len(kf_vib_test[i])
            conf.append(confusion_matrix(ground_truth , out))#, labels=[0, 1]))
            #accuracy
            accuracy.append(accuracy_score(ground_truth, out))
            #print("kFold",i)
            #print(conf[i],accuracy[i])
            #imshow
            #print ("==================================")
            print(j,n,i)
           
        print("ALL , n_components = ",j)
        print(sum(conf),np.mean(accuracy))
        n_ALL_accuracy.append(np.mean(accuracy))
        print ("==================================")
    plt.plot(n_ALL_accuracy[(j-start)*N:(j-start+1)*N])
    plt.savefig("saveplot\\"+str(j)+" components accuracy.png",dpi = 500)
    plt.show()
    #print(n_ALL_accuracy)
"""
for i in range(len(n_ALL_accuracy)//N):
    plt.plot(n_ALL_accuracy[i*N:(i+1)*N])
    plt.savefig("saveplot\\"+str(i+start)+" components accuracy.png",dpi = 500)
    plt.show()
"""

ALL_accuracy = [ np.mean(n_ALL_accuracy[i*N:(i+1)*N]) for i in range(len(n_ALL_accuracy)//N)]
plt.plot(range(start,end),ALL_accuracy)
plt.savefig("saveplot\\All components accuracy.png",dpi = 500)
plt.show()


"""
#hidden_states = modelvib.predict(vib_mfcc)
#print (modelvib.score(vib_mfcc))
#print (modelvib.startprob_)
#print (modelvib.transmat_)
#print (modelvib.means_)
#print (modelvib.covars_)
#print(hidden_states)
"""



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