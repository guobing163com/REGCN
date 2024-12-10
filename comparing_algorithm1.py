# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 09:51:02 2020

@author: Administrator
"""
#运行于python3.7
#实现GN算法，谱聚类算法
from cdlib import algorithms, viz
import networkx as nx
import igraph as ig
from read_gml import *
from utils import load_data, accuracy
import numpy as np
from modularity import *
from GN import *
from SpectralClustering import *
from sklearn.metrics import accuracy_score

#G = nx.karate_club_graph()
adj, features, labels, idx, G= load_data()
#G=nx.read_gml("data/dolphins/dolphins.gml")

#coms = algorithms.louvain(G)#可用
coms = GN_partition(G)#GN算法
#coms = SpectralClustering.partition(G, 12)  # 谱聚类

pos = nx.spring_layout(G)
#print(coms.communities)

preds=coms[0]
print(labels)
print(preds)
print(idx)
print('模块数=',len(set(preds)))
print('模块性=',coms[1])

NMI = metrics.normalized_mutual_info_score(labels, preds)
print('NMI = ',NMI)
ARI = metrics.adjusted_rand_score(labels, preds)
print('ARI = ',ARI)
ACC = accuracy_score(labels, preds)#计算准确性
print('ACC = ',ACC)
#viz.plot_network_clusters(G, coms, pos)#画聚类结果
#viz.plot_community_graph(G, coms)#画社区发现结果