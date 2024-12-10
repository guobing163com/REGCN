# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:21:15 2020

@author: Administrator
"""
import time
import argparse
import numpy as np
import networkx as nx

import torch
import torch.nn.functional as F
import torch.optim as optim
import random

def settrain(G,idx,m,idx_center_node):

    index_center_node=[]
    idx_train=[]
    idx_train_list=[]
    labels1=[]
    for i in range(len(idx_center_node)):
        index_center_node.append(idx.tolist().index(idx_center_node[i]))
    print('种子节点位置：',index_center_node)
    A=np.array(nx.adjacency_matrix(G).todense())#获取邻接矩阵A
    #print('A=',A)
    for i in range(len(index_center_node)):#将中心节点1邻域节点加入训练集中
        idx_train_list.append([j for j,x in enumerate(A[index_center_node[i]]) if x == 1 ])
    #print(idx_train_list)
    for i in range(len(idx)):
        count=0
        for j in range(m):
            if i in idx_train_list[j]:
                count+=1;
        if count>=2:#将有重叠的节点去掉
            for j in range(m):
                if i in idx_train_list[j]:
                    idx_train_list[j].remove(i)
    #print(idx_train_list)
    for i in range (m):#最后将初始中心节点加入训练集中，生成标签
        idx_train_list[i].append(index_center_node[i])
        labels=[i for j in range(len(idx_train_list[i]))]
        idx_train=idx_train+idx_train_list[i]
        labels1 = labels1+labels
    print('idx_train=',idx_train)
    print('labels1=',labels1)
    #idx_train = [0,1, 5, 7, 9, 11, 13, 17, 23, 25, 27, 29, 31, 32, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30,33]
    #labels2=[1 for i in range(14)]
    #labels3 = [0 for i in range(13)]
    #labels1 = labels2+labels3
    #print(labels1)
    return idx_train,labels1
# =============================================================================
# def updatenode(G,comm,idx,upNum):#upNum表示默认类中心点数的倍数
#     #针对每一个划分comm，计算SLP指标，选择权重最大的点作为该comm中心节点
#     init_node_list=[]
#     A=np.array(nx.adjacency_matrix(G).todense())
#     #print(A)
#     print('comm=',comm)
#     for list1 in comm:#针对每一个comm，计算comm的类中心节点
#         g=G.subgraph(list1)
#         
#         A2=np.array(nx.adjacency_matrix(g).todense())
#         A1=np.zeros((len(list1),len(list1)))
#         print('list1=',list1)
#         print('g.nodes=',g.nodes())
#         for i in range(len(list1)):#构造comm的邻接矩阵A1
#             for j in range(i,len(list1)):
#                 if A[idx.tolist().index(list1[i])][idx.tolist().index(list1[j])]==1:
#                     A1[i][j]=1
#                     A1[j][i]=1
#         print('A1=',A1)
#         print('A2=',A2)
#         #A2=np.dot(A1,A1)
#         #A3=np.dot(A2,A1)
#         LS1 =A2
#         #LS1 = 0.6*A1+0.35*A2+0.05*A3
#         #LS1 = 0.8*A1+0.15*np.dot(A1,A1)+0.05*np.dot(np.dot(A1,A1),A1)
#         #LS1 = 0.8*A1+0.15*np.dot(A1,A1)+0.05*np.dot(np.dot(A1,A1),A1)
#         node_num=len(list1)
#         weight = np.zeros(node_num)
# 
#         for i in range(node_num):
#             for j in range(node_num):
#                 if j!=i:
#                     weight[i]+=LS1[i][j]
#         #print('weight = ',weight)
#         #c = np.argmax(weight)
#         #print(weight)
#         #c_list1 = np.argwhere(weight == np.amax(weight))#列出所有最大值的索引
# 
#         #print('c=',c_list)
#         #index = random.randint(0,len(c_list)-1)#随机选一个最大值对应的索引
#         #print(weight)
#         #print(index)
#         if upNum==1:
#             c_list1 = np.argwhere(weight == np.amax(weight))#列出所有最大值的索引
#             c_list = c_list1.flatten().tolist()#转换为list
#             index = random.randint(0,len(c_list)-1)#随机选一个最大值对应的索引
#             init_node_list.append(list1[c_list[index]])
#         else:                
#             c_list1 = np.argpartition(weight,-upNum)[-upNum:]#列出weight中top upNum的索引
#             c_list = c_list1.flatten().tolist()#转换为list
#             for index in c_list:
#                 init_node_list.append(list1[index])
#         #print(init_node_list)
#         
#         #print(init_node_list)
#     return init_node_list
# =============================================================================

def updatenode(G,comm,idx,upNum):#upNum表示默认类中心点数的倍数
    #针对每一个划分comm，计算SLP指标，选择权重最大的点作为该comm中心节点
    init_node_list=[]
    #A=np.array(nx.adjacency_matrix(G).todense())
    #print(A)
    #print('comm=',comm)
    for list1 in comm:#针对每一个comm，计算comm的类中心节点
        g=G.subgraph(list1)
        
        A1=np.array(nx.adjacency_matrix(g).todense())
        list_nodes=list(g.nodes())#g.nodes()不是list类型，必须先将其转为list类型
        #A2=np.dot(A1,A1)
        #A3=np.dot(A2,A1)
        LS1 = A1
        #LS1 = 0.6*A1+0.35*A2+0.05*A3
        #LS1 = 0.4*A1+0.3*A2+0.3*A3
        #LS1 = 0.6*A1+0.4*A2
        #print('list1=',list1)
        #print('list_nodes=',list_nodes)
        node_num=len(list1)
        weight = np.zeros(node_num)

        for i in range(node_num):
            for j in range(node_num):
                if j!=i:
                    weight[i]+=LS1[i][j]
                    
        #print('weight=',weight)
        if upNum==1:
            c_list1 = np.argwhere(weight == np.amax(weight))#列出所有最大值的索引
            #print('c_list1=',c_list1)
            c_list = c_list1.flatten().tolist()#转换为list
            #print('c_list=',c_list)
            index = random.randint(0,len(c_list)-1)#随机选一个最大值对应的索引
            init_node_list.append(list_nodes[c_list[index]])
        else:                
            c_list1 = np.argpartition(weight,-upNum)[-upNum:]#列出weight中top upNum的索引
            c_list = c_list1.flatten().tolist()#转换为list
            for index in c_list:
                init_node_list.append(list_nodes[index])
        
        #print(init_node_list)
    return init_node_list