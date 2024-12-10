# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 08:46:52 2020

@author: Administrator
"""
import networkx as nx
from sklearn import metrics
import numpy as np

def modularity(comm,G):

   #边的个数
   edges=G.edges()
   m=len(edges)
   #print 'm',m

   #每个节点的度
   du=G.degree()

   #通过节点对计算
   ret2=0.0
   for c in comm:
       #首先计算出社区c中的边的个数
       #这样计算出来的边数是实际边数的2倍
       bian=0
       for x in c:
           for y in c:
               #边都是前小后大的
               #不能交换x，y，因为都是循环变量
               if x<=y:
                   if (x,y) in edges:
                       bian=bian+1
               else:
                   if (y,x) in edges:
                       bian=bian+1
       #社区c内的节点度的和
       duHe=0
       for x in c:
           duHe=duHe+du[x]
       tmp=bian*1.0/(2*m)-(duHe*1.0/(2*m))*(duHe*1.0/(2*m))
       #print 'bian',bian,'tmp',tmp
       ret2=ret2+tmp
   return ret2

if __name__ == "__main__":
    g=nx.Graph()
    g.add_nodes_from(range(14))
    print(g.nodes())
    g.add_edges_from([(0,1),(0,2),(0,3),(1,2),(1,3),(2,4),(4,5),(4,7),(5,6),(5,7),(5,8),
                      (6,8),(7,8),(7,10),(8,9),(9,10),(9,12),(9,13),(10,11),(10,12),(11,12),(11,13),(12,13)])
    print(g.edges())
    comm=[[0,1,2,3],[4,5,6,7,8],[9,10,11,12,13]]
    Q = modularity(comm,g)
    print(Q)
    A=np.array(nx.adjacency_matrix(g).todense())#获取邻接矩阵A
    #LS = A + 0.4*A*A + 0.4*A*A*A
    LS1 = A+0.4*np.dot(A,A)+0.4*np.dot(np.dot(A,A),A)
    print(A)
    #print(LS)
    print(LS1)
# =============================================================================
#     labels_true = [0, 0, 0, 1, 1, 1]
#     labels_pred = [1, 1, 1, 0, 0, 0]
#     print(metrics.normalized_mutual_info_score(labels_true, labels_pred))
# =============================================================================
