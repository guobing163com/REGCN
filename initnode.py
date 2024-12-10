# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:07:02 2020

@author: Administrator
"""
import networkx as nx
from sklearn import metrics
import numpy as np
from utils import load_data, accuracy
import random
from plot import *
# =============================================================================
# def initnode_LCC(g,m):
#     
#     idx=list(g.node())
#     #print(list(idx))
#     nodeDegree = g.degree()#计算所有节点的度
#     #print(nodeDegree)
#     nodeDegreeList = list(nodeDegree)#转为list格式(节点,节点的度)
#     nodeDegreeList.sort(key=lambda x:x[1],reverse=True)#按节点的度降序排序
#     print('nodeDegreeList=',nodeDegreeList)
#     #diameter = nx.diameter(g)#计算网络直径
#     #diameter = 18
#     #print('直径=',diameter)
#     sc=[]#存放节点度与距离的乘积(节点号，最小距离*度,最小距离，度)
#     nodeMaxDegree = nodeDegreeList[0][0]#找出度最大的节点
#     #print(nodeMaxDegree)
#     
#     #选出节点之间最大距离
#     maxdistance_dict = nx.shortest_path_length(g,source = nodeMaxDegree)
#     d = sorted(maxdistance_dict.items(),key=lambda x:x[1],reverse=True)
#     maxdistance = d[0][1]
#     #print(maxdistance)
#     sc.append((nodeMaxDegree,nodeDegreeList[0][1]*maxdistance,maxdistance,nodeDegreeList[0][1]))#将度最大节点的度与（直径）的乘积放入sc中
#     
#     #print(sc)
#     init_node = []
#     init_node1=[]
#     init_node.append(nodeMaxDegree)
#     init_node1.append((nodeMaxDegree,nodeDegreeList[0][1]))
#     
#     #for k in range(100):#计算前m个节点sc值
#     for k in range(len(idx)-1):#计算全部节点SC值
#         sc1=[]
#         for i in range(len(idx)):#循环将其余点的度与其余点和比自己度大点之间的最小距离的成绩放入sc中
#             if nodeDegreeList[i][0] in init_node:#已选上的点不参与计算，忽略跳过
#                 continue
#             mindistanc = maxdistance
#             for j in range(len(init_node)):#计算与比自己度大节点的距离，选择最小距离
#                 try:
#                     distance = nx.dijkstra_path_length(g,nodeDegreeList[i][0],init_node[j])#计算两点之间的最短距离
#                 except:#如果有异常，说明有不连通的点，此时计不连通点之间的距离为0
#                     distance = 0
#                 if distance < mindistanc:
#                     mindistanc = distance
#             #print(i)
#             sc1.append((nodeDegreeList[i][0],mindistanc*nodeDegreeList[i][1],mindistanc,nodeDegreeList[i][1]))
#         #print(sc1)
#         sc1.sort(key=lambda x:x[1],reverse=True)#按乘积降序排序
#         #print(sc1)
#         sc.append((sc1[0][0],sc1[0][1],sc1[0][2],sc1[0][3]))
#         init_node.append(sc1[0][0])
#         init_node1.append((sc1[0][0],sc1[0][3]))
#     print('sc = ',sc)
#     plot(sc)
#     return init_node[0:m]
# 
# def initnode_LCC1(g,m):
#     
#     idx=list(g.node())
#     #print(list(idx))
#     nodeDegree = g.degree()#计算所有节点的度
#     #print(nodeDegree)
#     nodeDegreeList = list(nodeDegree)#转为list格式(节点,节点的度)
#     nodeDegreeList.sort(key=lambda x:x[1],reverse=True)#按节点的度降序排序
#     print('nodeDegreeList=',nodeDegreeList)
#     #diameter = nx.diameter(g)#计算网络直径
#     #diameter = 18
#     #print('直径=',diameter)
#     sc=[]#存放节点度与距离的乘积(节点号，最小距离*度,最小距离，度)
#     nodeMaxDegree = nodeDegreeList[0][0]#找出度最大的节点
#     #print(nodeMaxDegree)
#     
#     #选出节点之间最大距离
#     maxdistance_dict = nx.shortest_path_length(g,source = nodeMaxDegree)
#     d = sorted(maxdistance_dict.items(),key=lambda x:x[1],reverse=True)
#     maxdistance = d[0][1]
#     #print(maxdistance)
#     sc.append((nodeMaxDegree,nodeDegreeList[0][1]*maxdistance,maxdistance,nodeDegreeList[0][1]))#将度最大节点的度与（直径）的乘积放入sc中
#     
#     #print(sc)
#     init_node = []
#     init_node1=[]
#     init_node.append(nodeMaxDegree)
#     init_node1.append((nodeMaxDegree,nodeDegreeList[0][1]))
#     
#     #for k in range(100):#计算前m个节点sc值
#     for k in range(len(idx)-1):#计算全部节点SC值
#         sc1=[]
#         for i in range(len(idx)):#循环将其余点的度与其余点和比自己度大点之间的最小距离的成绩放入sc中
#             if nodeDegreeList[i][0] in init_node:#已选上的点不参与计算，忽略跳过
#                 continue
#             mindistanc = maxdistance
#             for j in range(len(init_node)):#计算与比自己度大节点的距离，选择最小距离
#                 if init_node1[j][1]>nodeDegreeList[i][1]:
#                     try:
#                         distance = nx.dijkstra_path_length(g,nodeDegreeList[i][0],init_node[j])#计算两点之间的最短距离
#                     except:#如果有异常，说明有不连通的点，此时计不连通点之间的距离为0
#                         distance = 0
#                     if distance < mindistanc:
#                         mindistanc = distance
#             #print(i)
#             sc1.append((nodeDegreeList[i][0],mindistanc*nodeDegreeList[i][1],mindistanc,nodeDegreeList[i][1]))
#         print(sc1)
#         sc1.sort(key=lambda x:x[1],reverse=True)#按乘积降序排序
#         print(sc1)
#         sc.append((sc1[0][0],sc1[0][1],sc1[0][2],sc1[0][3]))
#         init_node.append(sc1[0][0])
#         init_node1.append((sc1[0][0],sc1[0][3]))
#     print('sc = ',sc)
#     plot(sc)
#     return init_node[0:m]
# =============================================================================
def initnode_LCC2(g,m):#快速查找中心节点
    
    idx=list(g.node())
    #print(list(idx))
    nodeDegree = g.degree()#计算所有节点的度
    #print(nodeDegree)
    nodeDegreeList = list(nodeDegree)#转为list格式(节点,节点的度)
    nodeDegreeList.sort(key=lambda x:x[1],reverse=True)#按节点的度降序排序
    #print('nodeDegreeList=',nodeDegreeList)
    #diameter = nx.diameter(g)#计算网络直径
    #diameter = 18
    #print('直径=',diameter)
    sc=[]#存放节点度与距离的乘积(节点号，最小距离*度,最小距离，度)
    nodeMaxDegree = nodeDegreeList[0][0]#找出度最大的节点
    #print(nodeMaxDegree)
    
    #选出节点之间最大距离
    maxdistance_dict = nx.shortest_path_length(g,source = nodeMaxDegree)
    d = sorted(maxdistance_dict.items(),key=lambda x:x[1],reverse=True)
    maxdistance = d[0][1]
    #print(maxdistance)
    sc.append((nodeMaxDegree,nodeDegreeList[0][1]*maxdistance,maxdistance,nodeDegreeList[0][1]))#将度最大节点的度与（直径）的乘积放入sc中
    
    #print(sc)
    init_node = []
    init_node1=[]
    init_node.append(nodeMaxDegree)
    init_node1.append((nodeMaxDegree,nodeDegreeList[0][1]))
    
    for i in range(200):#此处数字为节点总数或者一部分节点数量，#循环将其余点的度与其余点和比自己度大点之间的最小距离的乘积放入sc中
    #for i in range(len(idx)):#循环将其余点的度与其余点和比自己度大点之间的最小距离的乘积放入sc中
        if nodeDegreeList[i][0] in init_node:#已选上的点不参与计算，忽略跳过
            continue
        mindistanc = maxdistance
        for j in range(len(init_node)):#计算与比自己度大节点的距离，选择最小距离
            if init_node1[j][1]>=nodeDegreeList[i][1]:
                try:
                    distance = nx.dijkstra_path_length(g,nodeDegreeList[i][0],init_node[j])#计算两点之间的最短距离
                except:#如果有异常，说明有不连通的点，此时计不连通点之间的距离为0
                    distance = 0
                if distance < mindistanc:
                    mindistanc = distance
        #print(i)
        sc.append((nodeDegreeList[i][0],mindistanc*nodeDegreeList[i][1],mindistanc,nodeDegreeList[i][1]))
        init_node.append(nodeDegreeList[i][0])
        init_node1.append((nodeDegreeList[i][0],nodeDegreeList[i][1]))
    #print(sc)
    sc.sort(key=lambda x:x[1],reverse=True)#按乘积降序排序
    #print(sc)
    #sc.append((sc1[0][0],sc1[0][1],sc1[0][2],sc1[0][3]))

    print('sc = ',sc)
    plot_log(sc)
    #plot_2(sc)
    end_init_node=[]
    for i in range(m):
        end_init_node.append(sc[i][0])
    return end_init_node
def initnode(g,m):#使用SLP计算中心节点
    idx=list(g.node())
    node_num = g.number_of_nodes()
    #print(list(idx))
    A=np.array(nx.adjacency_matrix(g).todense())#获取邻接矩阵A
    #print(A)
    #LS = A + 0.4*A*A + 0.4*A*A*A
    #A2=np.linalg.inv(np.identity(node_num) - 0.3*A)-np.identity(node_num)
    A2=np.dot(A,A)
    print("A2计算完毕")
    A3=np.dot(A2,A)
    LS = A3
    LS = 0.6*A+0.35*A2+0.05*A3
    print("LS矩阵计算完毕")
    #print(A)
    #print(LS)
    #print('LS = ')
    #print(LS)
    
    weight = np.zeros(node_num)

# =============================================================================
#     for i in range(node_num):#计算weight数组
#         for j in range(node_num):
#             if j!=i:
#                 weight[i]+=LS[i][j]
# =============================================================================
    weight = np.sum(LS,axis=1)#按行求和  
    print('weight计算完毕')
    
    init_node_list=[]
    c = np.argmax(weight)#计算第一个类中心节点
    init_node_list.append(c)
    #print('centrnode = ',init_node_list)
    for h in range(m-1):
        ph=np.zeros(node_num)
        for i in range(node_num):
            maxw=0
            for j in range(len(init_node_list)):
                if LS[i][init_node_list[j]]>maxw:
                    maxw=LS[i][init_node_list[j]]
                    #print('maxw=',LS[i][init_node_list[j]],'i=',i,'j=',j)
                
            ph[i]=weight[i]/(maxw+1)
            for j in range(len(init_node_list)):#将已选类中心点对应位置置为0
                ph[init_node_list[j]]=0
        #print('ph = ',ph)
        c = np.argmax(ph)
        init_node_list.append(c)
    #print(init_node_list)
    #init_node_list=[91,105,55,107,43,56,106,41,83,108,50,90]
    #init_node_list=[47,35,37,52,66,26,67,97,10,80,96,68]
    #print(init_node_list)
    init_node=[]
    for i in range(len(init_node_list)):
        init_node.append(idx[init_node_list[i]])#找出节点的id号
    return init_node
def initnode1(g,m):
    
    A=np.array(nx.adjacency_matrix(g).todense())#获取邻接矩阵A
    #LS = A + 0.4*A*A + 0.4*A*A*A
    LS = 0.8*A+0.15*np.dot(A,A)+0.05*np.dot(np.dot(A,A),A)
    for i in range(len(LS)):
        LS[i][i]=0
    #print(A)
    #print(LS)
    #print('LS = ')
    #print(LS)
    node_num = g.number_of_nodes()
    weight = np.zeros(node_num)

# =============================================================================
#     for i in range(node_num):
#         for j in range(node_num):
#             if j!=i:
#                 weight[i]+=LS[i][j]
# =============================================================================
    for i in range(node_num):
        for j in range(node_num):
            weight[i]+=LS[i][j]
    #print('weight = ',weight)
    
    init_node_list=[]
    c = np.argmax(weight)
    init_node_list.append(c)
    #print('centrnode = ',init_node_list)
    for h in range(m-1):
        ph=np.zeros(node_num)
        for i in range(node_num):
            maxw=0
            value=[]
            for j in range(len(init_node_list)):
                #maxw+=LS[i][init_node_list[j]]
                value.append(LS[i][init_node_list[j]])

            #ph[i]=weight[i]/(maxw+1)
            ph[i]=weight[i]/(np.var(value)+np.sum(value)+0.1)
            for j in range(len(init_node_list)):#将已选类中心点对应位置置为0
                ph[init_node_list[j]]=0
        #print('ph = ',ph)
        c = np.argmax(ph)
        init_node_list.append(c)
        print(init_node_list)
    
    return init_node_list
if __name__ == "__main__":
    g=nx.Graph()
    node_list=[10,11,12,0,1,4,5,6,13,2,3,7,8,9]
    g.add_nodes_from(node_list)
    #print(g.nodes())
    g.add_edges_from([(0,1),(0,2),(0,3),(1,2),(1,3),(2,4),(4,5),(4,7),(5,6),(5,7),(5,8),
                      (6,8),(7,8),(7,10),(8,9),(9,10),(9,12),(9,13),(10,11),(10,12),(11,12),(11,13),(12,13)])
    #print(g.edges())
    comm=[[0,1,2,3],[4,5,6,7,8],[9,10,11,12,13]]
    #init_node_list = initnode(g,3)
    #print(init_node_list)
    #adj, features, labels, idx, G= load_data()
    init_node_list = initnode(g,3)
    print(init_node_list)
    #print(labels)
    label = [1,2,3,4,5,6]
    label = torch.LongTensor(label)
    train = [1,3]
    labels1=[i for i in range(m)]
    labels2 = [i for i in range(m)]
    labels1 = labels1+labels1
    labels1.sort()
    print(labels1)
    npnode=np.array(node_list)
    index = np.argpartition(npnode,-2)[-2:]
    print(index)
    
    #print(label[train])
    #print( random.randint(1,2) )
