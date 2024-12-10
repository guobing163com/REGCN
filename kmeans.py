# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 17:24:51 2022

@author: Administrator
"""

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN
from sklearn import metrics

from modularity import *
from initnode import *
from updatenode import *
import random
from sklearn.cluster import KMeans
import csv

# Load data
adj, features, labels, idx, G= load_data()
#labels_tensor= torch.LongTensor(labels)
k=6
for i in range(20):
    result = KMeans(n_clusters=k).fit_predict(features)
    print("标准标签",labels)
    print("预测标签",result)
    NMI = metrics.normalized_mutual_info_score(labels, result)
    print('NMI = ',NMI)
    ARI = metrics.adjusted_rand_score(labels, result)
    print('ARI = ',ARI)
    with open("kmeanResult.csv","a+",newline='') as csvfile: 
        writer = csv.writer(csvfile)
         
        #将实验参数和实验结果写入csv文件
        #writer.writerow(["数据集","社区个数","NMI","ARI"])
        writer.writerow(["citeseer",k,NMI,ARI])
        csvfile.close()