# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 13:21:55 2024

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
from munkres import Munkres


def accuracy1(labels, preds): #其它对比算法使用该函数计算ACC指标

    preds = best_map(labels, preds)
    ACC = accuracy_score(labels, preds)#计算准确性

    return ACC

def best_map(L1,L2): #使用Kuhn-Munkres(KM)算法将聚类后的预测标签映射为真实标签
	#L1 should be the labels and L2 should be the clustering number we got
	Label1 = np.unique(L1)       # 去除重复的元素，由小大大排列
	nClass1 = len(Label1)        # 标签的大小
	Label2 = np.unique(L2)       
	nClass2 = len(Label2)
	nClass = np.maximum(nClass1,nClass2)
	G = np.zeros((nClass,nClass))
	for i in range(nClass1):
		ind_cla1 = L1 == Label1[i]
		ind_cla1 = ind_cla1.astype(float)
		for j in range(nClass2):
			ind_cla2 = L2 == Label2[j]
			ind_cla2 = ind_cla2.astype(float)
			G[i,j] = np.sum(ind_cla2 * ind_cla1)
	m = Munkres()
	index = m.compute(-G.T)
	index = np.array(index)
	c = index[:,1]
	newL2 = np.zeros(L2.shape)
	for i in range(nClass2):
		newL2[L2 == Label2[i]] = Label1[c[i]]
	return newL2

labels = np.array([ 0,  5,  7, 11,  9,  1,  2,  3,  6,  4, 10,  8,  2,  7,  5, 11,  0,  9,
         6, 10,  3,  8,  1,  4,  1,  7, 11,  9,  0, 10,  5,  2,  4,  8,  6,  1,
         0,  9,  1,  5,  9,  9, 10, 11,  3, 11,  0, 10,  2,  6,  1,  7,  4,  6,
         5,  1,  0,  6,  0,  9,  3,  4,  9,  1, 11,  0,  9,  1,  9, 11,  7,  1,
         5,  2, 10,  4, 11,  4,  6,  4, 11,  2,  2,  5,  1,  6,  9, 10,  1,  3,
         8, 11,  3,  6,  4,  7,  3,  5,  6,  2,  2,  2,  3,  7,  0,  5, 11,  7,
         9,  9, 11,  8,  4,  2,  0])
preds = np.array([6.0, 1.0, 0.0, 4.0, 3.0, 2.0, 0.0, 1.0, 7.0, 5.0, 8.0, 8.0, 0.0, 0.0, 1.0, 4.0, 6.0, 3.0, 7.0, 2.0, 11.0, 9.0, 2.0, 5.0, 2.0, 10.0, 4.0, 3.0, 6.0, 2.0, 1.0, 0.0, 5.0, 0.0, 7.0, 2.0, 6.0, 3.0, 2.0, 1.0, 3.0, 3.0, 2.0, 4.0, 1.0, 4.0, 6.0, 8.0, 0.0, 2.0, 2.0, 0.0, 5.0, 7.0, 1.0, 2.0, 6.0, 7.0, 7.0, 3.0, 1.0, 5.0, 3.0, 2.0, 4.0, 6.0, 3.0, 2.0, 3.0, 4.0, 0.0, 2.0, 1.0, 0.0, 8.0, 5.0, 4.0, 5.0, 7.0, 5.0, 4.0, 0.0, 0.0, 1.0, 2.0, 7.0, 3.0, 8.0, 2.0, 1.0, 3.0, 4.0, 1.0, 7.0, 5.0, 0.0, 1.0, 1.0, 8.0, 0.0, 0.0, 0.0, 1.0, 0.0, 6.0, 1.0, 4.0, 0.0, 3.0, 3.0, 4.0, 3.0, 5.0, 0.0, 6.0])
new_preds = np.where(preds>=3,2,preds)
    
ACC = accuracy1(labels, new_preds)#计算准确性
print('ACC = ',ACC)