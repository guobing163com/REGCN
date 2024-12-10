# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 09:51:02 2020

@author: Administrator
"""

from cdlib import algorithms, viz
import networkx as nx
import igraph as ig
from read_gml import *
from utils import load_data, accuracy,accuracy1
import numpy as np
from modularity import *
from GN import*
from SpectralClustering import *
from sklearn.metrics import accuracy_score

#G = nx.karate_club_graph()
adj, features, labels, idx, G= load_data()
# =============================================================================
# #G=nx.read_gml("data/dolphins/dolphins.gml")
# ##coms = algorithms.eigenvector(G)#可用
# ##coms = algorithms.der(G, 3, .00001, 50)#可用
# ##coms = algorithms.girvan_newman(G, level=3)#可用
# #coms = algorithms.greedy_modularity(G)#可用
# 
# ##coms = algorithms.markov_clustering(G, max_loop=1000)#可用
# ##coms = algorithms.spinglass(G)#可用
# ##coms = algorithms.walktrap(G)#可用
# #coms = algorithms.louvain(G)
# =============================================================================

#coms = algorithms.async_fluid(G,k=20)#可用FluidC
#coms = algorithms.em(G, k=20)#可用EM
#coms = algorithms.label_propagation(G)#可用LPA
coms = algorithms.louvain(G)#可用BGLL
    
pos = nx.spring_layout(G)
print(coms.communities)
print('模块数=',len(coms.communities))
preds=np.zeros(len(idx))
class_id=0
for coms1 in coms.communities:
    for i in coms1:
        preds[np.where(idx==i)]=class_id
    class_id+=1
print(labels)
print(preds.tolist())
#print(idx)

Q = modularity(coms.communities,G)
print('模块性=',Q)
NMI = metrics.normalized_mutual_info_score(labels, preds)
print('NMI = ',NMI)
ARI = metrics.adjusted_rand_score(labels, preds)
print('ARI = ',ARI)
ACC = accuracy_score(labels, preds)#计算准确性
print('ACC = ',ACC)
#ACC = accuracy1(labels, preds)#计算准确性
#print('ACC = ',ACC)
#viz.plot_network_clusters(G, coms, pos)#画聚类结果
#viz.plot_community_graph(G, coms)#画社区发现结果