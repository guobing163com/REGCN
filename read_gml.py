# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 17:34:24 2020

@author: Administrator
"""

import networkx as nx
import copy
import numpy as np
 
# 抽取gml中的数据
# networkx可以直接通过函数从gml文件中读出数据
# 可将gml数据转为.cites和.content数据
def read_gml(data):#此函数读取标准gml数据，文件中包括id,label,value
    G = nx.read_gml(data)
    
    #print(G.node['SouthernCalifornia']['value'])
    nodes = []
    edges = []
    nodes_id = dict()
    nodes_label = dict()
    nodes_value = dict()
    edges_id = []
    for id, label in enumerate(G.nodes()):
        nodes_id[label] = id
        nodes_label[id] = label
        nodes.append(id)
    for label in G.nodes():
        nodes_value[label] = G.node[label]['value']
    print(nodes_id)
    #print(nodes_id[1])
    for (v0, v1) in G.edges():
        #print(v1)
        temp = [nodes_id[v0], nodes_id[v1]]
        edges.append(temp)
    edges_id = copy.deepcopy(edges)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    #print(nodes_id)
    #print(nodes_label)
    #print(nodes_value)
    return G, nodes_id, edges_id, nodes_label,nodes_value
def read_gml1(data):#此函数读取不标准gml,文件中包括id和label,而value值在label中，用逗号隔开。例如netscience.gml
    G = nx.read_gml(data)
    
    #print(G.node['SouthernCalifornia']['value'])
    nodes = []
    edges = []
    nodes_id = dict()
    nodes_label = dict()
    nodes_value = dict()
    edges_id = []
    for id, label in enumerate(G.nodes()):
        label0=label.split(',')[0]
        nodes_id[label0] = id
        nodes_label[id] = label0
        nodes.append(id)
    for label in G.nodes():
        try:
            label0=label.split(',')[0]
            #print(label0)
            value=label.split(',')[1].strip()
            nodes_value[label0] = value
        except:
            print("error")
    #print(nodes_id)
    for (v0, v1) in G.edges():
        v00=v0.split(',')[0]
        v11=v1.split(',')[0]
        temp = [nodes_id[v00], nodes_id[v11]]
        edges.append(temp)
    edges_id = copy.deepcopy(edges)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    #print(nodes_id)
    #print(nodes_label)
    #print(nodes_value)
    return G, nodes_id, edges_id, nodes_label,nodes_value 
 
# 保存子图边集
def save_cites(data, file_name):
    f = open(file_name, 'w')
    temp = ''
    for item in data:
        temp += str(item[0]) + '\t' + str(item[1])
        temp += '\n'
    f.write(temp)
    f.close()
def save_content(nodes_value,nodes_id,file_name):
    sampleNum = len(nodes_value)
    contentMatrix=np.empty((sampleNum,sampleNum+2),dtype=object)
    contentMatrix[:,1:-1]=np.eye(sampleNum)
    key=list(nodes_value.keys())
    value=list(nodes_value.values())
    for i in range(len(nodes_value)):
        contentMatrix[i][0]=nodes_id[key[i]]
        contentMatrix[i][sampleNum+1]=value[i]
    np.random.seed(10)
    np.random.shuffle(contentMatrix)
    np.savetxt(file_name, contentMatrix,fmt='%s',delimiter='\t')
    print('属性矩阵生成，已保存。')
 
if __name__ == "__main__":
    #G, nodes_id, edges_id, nodes_label,nodes_value = read_gml('data/football/football.gml')
    #G, nodes_id, edges_id, nodes_label,nodes_value = read_gml1('data/netscience/netscience.gml')
    #print(edges_id)
    #save_cites(edges_id, 'data/netscience/netscience.cites.txt')
    #save_content(nodes_value,nodes_id,'data/netscience/netscience.content.txt')
# =============================================================================
#     label="PEREZVICENTE, C"
#     label0=label.split(',')[0]
#     print(label0)
#     value=label.split(',')[1].strip()
#     print(value)
# =============================================================================
    sampleNum = 4941
    contentMatrix=np.empty((sampleNum,sampleNum+2),dtype=object)
    contentMatrix[:,1:-1]=np.eye(sampleNum)
    for i in range(sampleNum):
        contentMatrix[i][0]=i
        contentMatrix[i][sampleNum+1]='none'
    #np.random.seed(10)
    #np.random.shuffle(contentMatrix)
    np.savetxt('data/power/power.content.txt', contentMatrix,fmt='%s',delimiter='\t')
    print('属性矩阵生成，已保存。')