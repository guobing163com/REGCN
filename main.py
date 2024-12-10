#该版本在python3.7下运行
#将utils.py中load_data（）函数文件路径和文件名修改一下，将data文件加内.cita文件和.content文件替换为实际数据，

from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, accuracy1
from models import GCN
from sklearn import metrics

from modularity import *
from initnode import *
from updatenode import *
import random

import csv


# Training settings设置各种参数值
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=60,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
m= 3 #初始种子节点数

# Load data
adj, features, labels, idx, G= load_data()
labels_tensor= torch.LongTensor(labels)
t_lag = 0
Q_max = 0
NMI_max = 0
ARI_max = 0
ACC_max = 0

    #updatenode(G,comm,idx)
def count_class(labels,idx,nodeList):
    #统计nodeList中节点真实属于几个类，比如3个点属于3各类，返回3，3个点属于2各类，返回2
    labelList = labels.numpy().tolist()
    idxList = idx.tolist()
    classList=[]
    for i in range(len(nodeList)):
        classList.append(labelList[idxList.index(nodeList[i])])
    classSet = set(classList)
    return classList,len(classSet)


def update_center(idx_train_init):#循环反复更新中心节点多次
    for i in range(1,20):#循环更新中心节点，可达到局部最优
    
        # 如果程序不禁止使用gpu且当前主机的gpu可用，arg.cuda就为True# 如果程序不禁止使用gpu且当前主机的gpu可用，arg.cuda就为True
        args = parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
            #print("cuda执行")
        # Model and optimizer
        #GCN(1433,16,7,0.5)
        #定义模型
        model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=m,
                    dropout=args.dropout)
        #定义优化器
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
        labels1=[i for i in range(m)]
        #idx_train,labels1 = settrain(G,idx,m,idx_center_node)
        idx_test = range(len(idx))
        idx_train = idx_train_init
        idx_test = torch.LongTensor(idx_test)
        labels1 = torch.LongTensor(labels1)
        
        #先用种子节点训练模型
        for epoch in range(args.epochs):
            model.train()#切换为训练模式
            optimizer.zero_grad()#先将梯度置为0
            output = model(features, adj)#用模型计算一次结果
            #print('output=',output)
            loss_train = F.nll_loss(output[idx_train], labels1)#计算损失函数
            acc_train = accuracy(output[idx_train], labels1)#计算准确性
            loss_train.backward()#反向传播，计算参数的梯度，也就是计算损失函数值对各个参数求导（斜率）
            optimizer.step()#更新参数，将反向传播计算的梯度*学习率加到上一次的参数中，得出最新的参数
        
        #得到初始划分   
        model.eval()#切换为评估模式
        output = model(features, adj)#用模型计算一次结果
        preds = output.max(1)[1].type_as(labels_tensor)
        preds_set=set(preds.numpy().tolist())
        preds_list = preds.numpy().tolist()
        comm=[]
        comm.clear()
        comm1=[]
        print('类个数',preds_set)
        print('类个数=',len(preds_set))
        for class_id in preds_set:
            comm1=[]
            for i in range(len(idx)):
                if preds_list[i]==class_id:
                    comm1.append(idx[i])
            comm.append(comm1)
        #print('com=',comm)
        Q = modularity(comm,G)
        print('模块性=',Q)
        NMI = metrics.normalized_mutual_info_score(labels, preds_list)
        print('NMI = ',NMI)
        ARI = metrics.adjusted_rand_score(labels, preds_list)
        print('ARI = ',ARI)
    
    
        idx_center_node = updatenode(G,comm,idx,1)#种子节点更新
        print('更新后种子节点为：',idx_center_node)
        listclass,numclass = count_class(labels,idx,idx_center_node)
        print('更新后种子节点类个数为：',numclass)
        idx_train_init=[]
        for i in range(len(idx_center_node)):#转换为.content中自然序号
            idx_train_init.append(idx.tolist().index(idx_center_node[i]))
        idx_train_init = torch.LongTensor(idx_train_init)
        print('idx_train_init0 = ',idx_train_init)
        
    return idx_center_node,idx_train_init

def construction_trainset_GCN(idx_center_node,idx_train_init,T):#产生平衡训练集，选择与ci最相似的t个节点作为训练集，
    #GCN预测结果中与ci值最近的认为最相似，训练集包括中心节点本身和t个最相似的点
    global t_lag
    global Q_max
    global NMI_max
    global ARI_max
    global ACC_max
    
    for t in range(1,T,5):#大循环，t表示平衡训练集的大小
        print('扩展标签节点数：',t)    
        # 如果程序不禁止使用gpu且当前主机的gpu可用，arg.cuda就为True# 如果程序不禁止使用gpu且当前主机的gpu可用，arg.cuda就为True
        args = parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
            #print("cuda执行")
        # Model and optimizer
        #定义模型
        model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=m,
                    dropout=args.dropout)
        #定义优化器
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
        labels1=[i for i in range(m)]
        #idx_train,labels1 = settrain(G,idx,m,idx_center_node)
        idx_test = range(len(idx))
        idx_train = idx_train_init
        #idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        labels1 = torch.LongTensor(labels1)
        
        #先用种子节点训练模型
        for epoch in range(args.epochs):
            model.train()#切换为训练模式
            optimizer.zero_grad()#先将梯度置为0
            output = model(features, adj)#用模型计算一次结果
            #print('output=',output)
            loss_train = F.nll_loss(output[idx_train], labels1)#计算损失函数
            acc_train = accuracy(output[idx_train], labels1)#计算准确性
            loss_train.backward()#反向传播，计算参数的梯度，也就是计算损失函数值对各个参数求导（斜率）
            optimizer.step()#更新参数，将反向传播计算的梯度*学习率加到上一次的参数中，得出最新的参数  
    
        #根据第1次训练结果，选取最可靠的前t个点最为平衡训练集
            
        center_node=[]#存放中心节点索引号（序号）
        for i in range(len(idx_center_node)):#转换为.content中自然序号
            center_node.append(idx.tolist().index(idx_center_node[i]))
        print('构建平衡集使用种子节点=',idx_center_node)
        model.eval()#切换为评估模式
        output = model(features, adj)#用模型计算一次结果,output为n*m矩阵
        #print(type(output))
        #print('output=',output)
        value_list = output.max(1)[0].tolist()#提取每行最大值，组成一维列表
        #print('value_list=',value_list)
        indices_list = output.max(1)[1].tolist()#提取每行最大值的列索引，组成一维列表
        #print('indices_list=',indices_list)
        class_num = len(set(indices_list))
        #print(class_num)
        label_seed_list = []
        labels1 = []
        for i in range(class_num):
            index_list = [j for j,x in enumerate(indices_list) if x==i]#将indices_list中各个类对应的序号提取出来
            #print('index_list=',index_list)
            index_np = np.array(index_list)#转换为np数组
            value_np = np.array(value_list)#转换为np数组
            #print('value_np=',value_np)
            value_np1 = value_np[index_np]#将序号对应的值提取出来
            #print('value_np1=',value_np1)
            for k in range(len(center_node)):#找出此index_list中包含哪个中心节点，将该中心节点序号赋值给index_center_node
                if center_node[k] in index_np:
                    index_center_node = center_node[k]
                    break
            #print('找到中心节点序号=',index_center_node)
            value_np1=abs(value_np1-value_np1[index_list.index(index_center_node)])#value_np每个元素减去中心节点的值，看哪些点与中心节点最近
            #print('value_np1=',value_np1)
            temp_array = np.column_stack([index_np,value_np1])#将序号和序号对应的预测值按列合并，方便一起排序
            #print('temp_array=',temp_array)
            temp_array=temp_array[np.lexsort(temp_array.T)]#按最后一列升序排序
            #print('temp_array=',temp_array)
            
            #将temp_array第1列前t+1个值提取出来，如果 temp_array[:,0][0:t]（包括中心节点本身）
            index_top_t = temp_array[:,0][0:t]
            #如果 temp_array[:,0][1:t+1]（不包括中心节点本身）
            #index_top_t = temp_array[:,0][1:t+1]
            #print(index_top_t)
            
            index_top_t = index_top_t.astype(int)#将index_top_t中所有元素类型转为int型
            #print(idx[index_top_t])
            label_seed_list = label_seed_list+index_top_t.tolist()#将每次循环（每个类）提取出来的top@t加入到label_seed_list中
            labels0=[i for j in range(len(index_top_t))]#生成对应的类标签
            labels1 = labels1+labels0#将生成对应的类标签加入到labels1中
        idx_train = label_seed_list
        
        #print('训练集=',idx[idx_train])
        idx_test = range(len(idx))
        idx_train = torch.LongTensor(idx_train)
        #idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        labels1 = torch.LongTensor(labels1)
        
        #用平衡训练集再次训练模型
        # 如果程序不禁止使用gpu且当前主机的gpu可用，arg.cuda就为True# 如果程序不禁止使用gpu且当前主机的gpu可用，arg.cuda就为True
        args = parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
            #print("cuda执行")
        # Model and optimizer
        #GCN(1433,16,7,0.5)
        #定义模型
        model1 = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=m,
                    dropout=args.dropout)
        #定义优化器
        optimizer1 = optim.Adam(model1.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
        #labels1=[i for i in range(m)]
        #idx_train,labels1 = settrain(G,idx,m,idx_center_node)
        idx_test = range(len(idx))
        idx_train = idx_train
        idx_test = torch.LongTensor(idx_test)
        labels1 = torch.LongTensor(labels1)
        
        #先训练模型
        for epoch in range(args.epochs):
            model1.train()#切换为训练模式
            optimizer1.zero_grad()#先将梯度置为0
            output = model1(features, adj)#用模型计算一次结果
            #print('output=',output)
            loss_train = F.nll_loss(output[idx_train], labels1)#计算损失函数
            acc_train = accuracy(output[idx_train], labels1)#计算准确性
            loss_train.backward()#反向传播，计算参数的梯度，也就是计算损失函数值对各个参数求导（斜率）
            optimizer1.step()#更新参数，将反向传播计算的梯度*学习率加到上一次的参数中，得出最新的参数
        
        #得到初始划分   
        model1.eval()#切换为评估模式
        output = model1(features, adj)#用模型计算一次结果
        preds = output.max(1)[1].type_as(labels_tensor)
        preds_set=set(preds.numpy().tolist())
        preds_list = preds.numpy().tolist()
        comm=[]
        comm.clear()
        comm1=[]
        print('类个数',preds_set)
        print('类个数=',len(preds_set))
# =============================================================================
#         for class_id in preds_set:
#             comm1=[]
#             for i in range(len(idx)):
#                 if preds_list[i]==class_id:
#                     comm1.append(idx[i])
#             comm.append(comm1)
#         #print(comm)
#         Q = modularity(comm,G)
#         print('模块性=',Q)
# =============================================================================
        Q=0
        NMI = metrics.normalized_mutual_info_score(labels, preds_list)
        print('NMI = ',NMI)
        ARI = metrics.adjusted_rand_score(labels, preds_list)
        print('ARI = ',ARI)
        #ACC = accuracy(output, labels)
        ACC = accuracy1(labels, preds_list)
        print('ACC = ',ACC)
        with open("ExperimentResults_trainNumber_loop.csv","a+",newline='') as csvfile: 
            writer = csv.writer(csvfile)
         
            #将实验参数和实验结果写入csv文件
            #writer.writerow(["数据集","节点数","边数","社区个数","类平均节点数","训练集节点数","NMI","ARI"])
            writer.writerow(["api",0,0,m,T,t,NMI,ARI,"都有"])
            csvfile.close()
    

        
        if NMI>NMI_max:
            t_lag = t
            Q_max = Q
            NMI_max = NMI
            ARI_max = ARI
            ACC_max = ACC
            
# =============================================================================
#             #当有进步的时候，更新一次中心节点，最后返回平衡训练集效果最好时对应的中心节点
#             idx_center_node_max = updatenode(G,comm,idx,1)#种子节点更新
#             print('平衡集后更新后种子节点为：',idx_center_node_max)
#             listclass,numclass = count_class(labels,idx,idx_center_node_max)
#             print('平衡集后更新后种子节点类个数为：',numclass)
#             idx_train_init_max=[]
#             for i in range(len(idx_center_node_max)):#转换为.content中自然序号
#                 idx_train_init_max.append(idx.tolist().index(idx_center_node_max[i]))
#     
#     return idx_center_node_max,idx_train_init_max
# =============================================================================

def construction_trainset_GCN_selftrain(idx_center_node,idx_train_init,T):#产生平衡训练集，选择与ci同类且z值最大的前t个节点作为训练集（t个一次性添加）
    #该方法来自self-training文章
    global t_lag
    global Q_max
    global NMI_max
    global ARI_max
    for t in range(1,5000,20):#大循环，t表示平衡训练集的大小
        print('扩展标签节点数：',t)    
        # 如果程序不禁止使用gpu且当前主机的gpu可用，arg.cuda就为True# 如果程序不禁止使用gpu且当前主机的gpu可用，arg.cuda就为True
        args = parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
            #print("cuda执行")
        # Model and optimizer
        #定义模型
        model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=m,
                    dropout=args.dropout)
        #定义优化器
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
        labels1=[i for i in range(m)]
        #idx_train,labels1 = settrain(G,idx,m,idx_center_node)
        idx_test = range(len(idx))
        idx_train = idx_train_init
        #idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        labels1 = torch.LongTensor(labels1)
        
        #先用种子节点训练模型
        for epoch in range(args.epochs):
            model.train()#切换为训练模式
            optimizer.zero_grad()#先将梯度置为0
            output = model(features, adj)#用模型计算一次结果
            #print('output=',output)
            loss_train = F.nll_loss(output[idx_train], labels1)#计算损失函数
            acc_train = accuracy(output[idx_train], labels1)#计算准确性
            loss_train.backward()#反向传播，计算参数的梯度，也就是计算损失函数值对各个参数求导（斜率）
            optimizer.step()#更新参数，将反向传播计算的梯度*学习率加到上一次的参数中，得出最新的参数  
    
        #根据第1次训练结果，选取最可靠的前t个点最为平衡训练集
            
        center_node=[]#存放中心节点索引号（序号）
        for i in range(len(idx_center_node)):#转换为.content中自然序号
            center_node.append(idx.tolist().index(idx_center_node[i]))
        print('构建平衡集使用种子节点=',idx_center_node)
        model.eval()#切换为评估模式
        output = model(features, adj)#用模型计算一次结果,output为n*m矩阵
        #print(type(output))
        #print('output=',output)
        value_list = output.max(1)[0].tolist()#提取每行最大值，组成一维列表
        #print('value_list=',value_list)
        indices_list = output.max(1)[1].tolist()#提取每行最大值的列索引，组成一维列表
        #print('indices_list=',indices_list)
        class_num = len(set(indices_list))
        #print(class_num)
        label_seed_list = []
        labels1 = []
        for i in range(class_num):
            index_list = [j for j,x in enumerate(indices_list) if x==i]#将indices_list中各个类对应的序号提取出来
            #print('index_list=',index_list)
            index_np = np.array(index_list)#转换为np数组
            value_np = np.array(value_list)#转换为np数组
            #print('value_np=',value_np)
            value_np1 = value_np[index_np]#将序号对应的值提取出来
            #print('value_np1=',value_np1)
            for k in range(len(center_node)):#找出此index_list中包含哪个中心节点，将该中心节点序号赋值给index_center_node
                if center_node[k] in index_np:
                    index_center_node = center_node[k]
                    break
            #print('找到中心节点序号=',index_center_node)
            #value_np1=abs(value_np1-value_np1[index_list.index(index_center_node)])#value_np每个元素减去中心节点的值，看哪些点与中心节点最近
            #print('value_np1=',value_np1)
            temp_array = np.column_stack([index_np,value_np1])#将序号和序号对应的预测值按列合并，方便一起排序
            #print('temp_array=',temp_array)
            temp_array=temp_array[np.lexsort(-temp_array.T)]#按最后一列排序 temp_array.T为升序，-temp_array.T为降序
            #print('temp_array=',temp_array)
            
            #将temp_array第1列前t+1个值提取出来，如果 temp_array[:,0][0:t]（包括中心节点本身）
            index_top_t = temp_array[:,0][0:t]
            #如果 temp_array[:,0][1:t+1]（不包括中心节点本身）
            #index_top_t = temp_array[:,0][1:t+1]
            #print(index_top_t)
            
            index_top_t = index_top_t.astype(int)#将index_top_t中所有元素类型转为int型
            #print(idx[index_top_t])
            label_seed_list = label_seed_list+index_top_t.tolist()#将每次循环（每个类）提取出来的top@t加入到label_seed_list中
            labels0=[i for j in range(len(index_top_t))]#生成对应的类标签
            labels1 = labels1+labels0#将生成对应的类标签加入到labels1中
        idx_train = label_seed_list
        
        #print('训练集=',idx[idx_train])
        idx_test = range(len(idx))
        idx_train = torch.LongTensor(idx_train)
        #idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        labels1 = torch.LongTensor(labels1)
        
        #用平衡训练集再次训练模型
        # 如果程序不禁止使用gpu且当前主机的gpu可用，arg.cuda就为True# 如果程序不禁止使用gpu且当前主机的gpu可用，arg.cuda就为True
        args = parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
            #print("cuda执行")
        # Model and optimizer
        #GCN(1433,16,7,0.5)
        #定义模型
        model1 = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=m,
                    dropout=args.dropout)
        #定义优化器
        optimizer1 = optim.Adam(model1.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
        #labels1=[i for i in range(m)]
        #idx_train,labels1 = settrain(G,idx,m,idx_center_node)
        idx_test = range(len(idx))
        idx_train = idx_train
        idx_test = torch.LongTensor(idx_test)
        labels1 = torch.LongTensor(labels1)
        
        #先训练模型
        for epoch in range(args.epochs):
            model1.train()#切换为训练模式
            optimizer1.zero_grad()#先将梯度置为0
            output = model1(features, adj)#用模型计算一次结果
            #print('output=',output)
            loss_train = F.nll_loss(output[idx_train], labels1)#计算损失函数
            acc_train = accuracy(output[idx_train], labels1)#计算准确性
            loss_train.backward()#反向传播，计算参数的梯度，也就是计算损失函数值对各个参数求导（斜率）
            optimizer1.step()#更新参数，将反向传播计算的梯度*学习率加到上一次的参数中，得出最新的参数
        
        #得到初始划分   
        model1.eval()#切换为评估模式
        output = model1(features, adj)#用模型计算一次结果
        preds = output.max(1)[1].type_as(labels_tensor)
        preds_set=set(preds.numpy().tolist())
        preds_list = preds.numpy().tolist()
        comm=[]
        comm.clear()
        comm1=[]
        print('类个数',preds_set)
        print('类个数=',len(preds_set))

        Q=0
        NMI = metrics.normalized_mutual_info_score(labels, preds_list)
        print('NMI = ',NMI)
        ARI = metrics.adjusted_rand_score(labels, preds_list)
        print('ARI = ',ARI)
        
        with open("ExperimentResults_trainNumber_loop1.csv","a+",newline='') as csvfile: 
            writer = csv.writer(csvfile)
         
            #将实验参数和实验结果写入csv文件
            #writer.writerow(["数据集","节点数","边数","社区个数","类平均节点数","训练集节点数","NMI","ARI"])
            writer.writerow(["pubmed",0,0,m,T,t,NMI,ARI,"无更新有构造"])
            csvfile.close()
def construction_trainset_GCN_selftrain1(idx_center_node,idx_train_init,T):#产生平衡训练集，选择与ci同类且z值最大的前t个节点作为训练集（一个一个添加）
    #该方法来自self-training文章
    global t_lag
    global Q_max
    global NMI_max
    global ARI_max
    
    labels1=[i for i in range(m)]
    #idx_train,labels1 = settrain(G,idx,m,idx_center_node)
    idx_test = range(len(idx))
    idx_train = idx_train_init
    #idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    labels1 = torch.LongTensor(labels1)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        #print("cuda执行")
    # Model and optimizer
    #定义模型
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=m,
                dropout=args.dropout)
    #定义优化器
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)        
    for t in range(1,5000,20):
        # 如果程序不禁止使用gpu且当前主机的gpu可用，arg.cuda就为True# 如果程序不禁止使用gpu且当前主机的gpu可用，arg.cuda就为True
  
        #先用种子节点训练模型
        for epoch in range(args.epochs):
            model.train()#切换为训练模式
            optimizer.zero_grad()#先将梯度置为0
            output = model(features, adj)#用模型计算一次结果
            #print('output=',output)
            loss_train = F.nll_loss(output[idx_train], labels1)#计算损失函数
            acc_train = accuracy(output[idx_train], labels1)#计算准确性
            loss_train.backward()#反向传播，计算参数的梯度，也就是计算损失函数值对各个参数求导（斜率）
            optimizer.step()#更新参数，将反向传播计算的梯度*学习率加到上一次的参数中，得出最新的参数  
            
        center_node=[]#存放中心节点索引号（序号）
        for i in range(len(idx_center_node)):#转换为.content中自然序号
            center_node.append(idx.tolist().index(idx_center_node[i]))
        #print('构建平衡集使用种子节点=',idx_center_node)
        model.eval()#切换为评估模式
        output = model(features, adj)#用模型计算一次结果,output为n*m矩阵
        #print(type(output))
        #print('output=',output)
        value_list = output.max(1)[0].tolist()#提取每行最大值，组成一维列表
        #print('value_list=',value_list)
        indices_list = output.max(1)[1].tolist()#提取每行最大值的列索引，组成一维列表
        #print('indices_list=',indices_list)
        class_num = len(set(indices_list))
        #print(class_num)
        label_seed_list = []
        labels1 = []
        for i in range(class_num):
            index_list = [j for j,x in enumerate(indices_list) if x==i]#将indices_list中各个类对应的序号提取出来
            #print('index_list=',index_list)
            index_np = np.array(index_list)#转换为np数组
            value_np = np.array(value_list)#转换为np数组
            #print('value_np=',value_np)
            value_np1 = value_np[index_np]#将序号对应的值提取出来
            #print('value_np1=',value_np1)
            for k in range(len(center_node)):#找出此index_list中包含哪个中心节点，将该中心节点序号赋值给index_center_node
                if center_node[k] in index_np:
                    index_center_node = center_node[k]
                    break
            #print('找到中心节点序号=',index_center_node)
            #value_np1=abs(value_np1-value_np1[index_list.index(index_center_node)])#value_np每个元素减去中心节点的值，看哪些点与中心节点最近
            #print('value_np1=',value_np1)
            temp_array = np.column_stack([index_np,value_np1])#将序号和序号对应的预测值按列合并，方便一起排序
            #print('temp_array=',temp_array)
            temp_array=temp_array[np.lexsort(-temp_array.T)]#按最后一列排序 temp_array.T为升序，-temp_array.T为降序
            #print('temp_array=',temp_array)
            
            #将temp_array第1列前t+1个值提取出来，如果 temp_array[:,0][0:t]（包括中心节点本身）
            index_top_t = temp_array[:,0][0:t]
            #如果 temp_array[:,0][1:t+1]（不包括中心节点本身）
            #index_top_t = temp_array[:,0][1:t+1]
            #print(index_top_t)
            
            index_top_t = index_top_t.astype(int)#将index_top_t中所有元素类型转为int型
            #print(idx[index_top_t])
            label_seed_list = label_seed_list+index_top_t.tolist()#将每次循环（每个类）提取出来的top@t加入到label_seed_list中
            labels0=[i for j in range(len(index_top_t))]#生成对应的类标签
            labels1 = labels1+labels0#将生成对应的类标签加入到labels1中
        idx_train = label_seed_list
        
        #print('训练集=',idx[idx_train])
        #print('标签序列',labels1)
        idx_test = range(len(idx))
        idx_train = torch.LongTensor(idx_train)
        #idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        labels1 = torch.LongTensor(labels1)
        #print('训练集=',idx[idx_train])
        #print('标签序列',labels1)
        

        #得到初始划分   
        model.eval()#切换为评估模式
        output = model(features, adj)#用模型计算一次结果
        preds = output.max(1)[1].type_as(labels_tensor)
        preds_set=set(preds.numpy().tolist())
        preds_list = preds.numpy().tolist()
        comm=[]
        comm.clear()
        comm1=[]
        print('类个数',preds_set)
        print('类个数=',len(preds_set))
    
        Q=0
        NMI = metrics.normalized_mutual_info_score(labels, preds_list)
        print('NMI = ',NMI)
        ARI = metrics.adjusted_rand_score(labels, preds_list)
        print('ARI = ',ARI)
        
        with open("ExperimentResults_trainNumber_loop2.csv","a+",newline='') as csvfile: 
            writer = csv.writer(csvfile)
         
            #将实验参数和实验结果写入csv文件
            #writer.writerow(["数据集","节点数","边数","社区个数","类平均节点数","训练集节点数","NMI","ARI","备注"])
            writer.writerow(["pubmed",0,0,m,T,t,NMI,ARI,"self-training"])
            csvfile.close()

#初始化种子节点
idx_train_init = []
#idx_center_node = [1164, 313, 8079, 6599, 3498, 5466, 4710, 1022, 7251, 8536, 7930, 4030, 5648, 2164, 479, 2423, 2600, 304, 2053, 1855]#api
#idx_center_node = [3309126,8366922,9742976]#pubmed
#idx_center_node = [4,14]#cornell
#idx_center_node1 = [34, 76, 19, 67, 3, 5, 2, 1, 6, 91, 109, 55]
#print('idx=',idx)

# =============================================================================
# #准确方法计算初始中心节点
# idx_center_node1 = initnode(G,m)
# print('准确方法计算种子节点为：',idx_center_node1)
# listclass,numclass = count_class(labels,idx,idx_center_node1)
# print('准确方法种子节点类个数为：',numclass)
# =============================================================================


#快速方法计算初始中心节点
idx_center_node = initnode_LCC2(G,m)
#idx_center_node= [67, 3, 7, 91, 43, 17, 104, 53, 5, 2, 88, 0]
#print('初始中心节点为：',idx_center_node)
#idx_center_node = [4,14]
# =============================================================================
# while(1):
#     idx_center_node = random.sample(list(idx), m)#随机抽取m个点作为中心节点
#     print('快速方法计算种子节点为：',idx_center_node)
#     listclass,numclass = count_class(labels,idx,idx_center_node)
#     print('种子节点类个数为：',numclass)
#     if(numclass>13):
#         break
# =============================================================================

for i in range(len(idx_center_node)):#转换为.content中自然序号
    idx_train_init.append(idx.tolist().index(idx_center_node[i]))
print('初始中心节点序号为：',idx_train_init)
idx_train_init = torch.LongTensor(idx_train_init)

#更新节点
#idx_center_node,idx_train_init = update_center(idx_train_init)
#print(idx_center_node)
#idx_center_node = [1,34]#karate
#idx_train_init = [0,33]#karate

#idx_center_node = [14, 13]#dolphins最优中心点
#idx_train_init = [37, 36]#dolphins最优中心点

#idx_center_node = [8,84,4]#polbooks最优中心点
#idx_train_init = [96,76,42]#polbooks最优中心点

#idx_center_node = [74, 39, 18, 111, 29, 25, 76, 90, 67, 48, 41, 98]#football最优中心点
#idx_train_init = [110,  82,  62,  32,  95,  72,  55,  11,  18, 114,   7,  76]#football最优中心点

#idx_center_node = [126, 837]#polblogs最优中心点
#idx_train_init = [896, 212]#polblogs最优中心点

#idx_center_node = [8012723, 11832527, 18776148]#pubmed最优中心点
#idx_train_init = [2012,  13853,  8928]#pubmed最优中心点

#idx_center_node = [20193,35,1365,39890,6213,2440,12182]#cora最优中心点
#idx_train_init = [1354, 163, 747,1527,565,344,453]#cora最优中心点

#idx_center_node = [847,1414,441,2049,364,1406]#citeseer最优中心点
#idx_train_init = [847,1414,441,2049,364,1406]#citeseer最优中心点

#idx_center_node = [191,164,2,145,88]#cornell最优中心点
#idx_train_init = [191,164,2,145,88]#cornell最优中心点

#T=int(len(idx)/m)
#print('最大训练集个数 T=',T)

#构建平衡训练集
#construction_trainset_GCN(idx_center_node,idx_train_init,T)
#construction_trainset_GCN_selftrain(idx_center_node,idx_train_init,T)

#idx_center_node,idx_train_init = update_center(idx_train_init_max)
#idx_center_node_max1,idx_train_init_max1 = construction_train_set(idx_center_node,idx_train_init,t)

# =============================================================================
# print('t_lag=',t_lag)
# print('Q_max=',Q_max)
# print('NMI_max=',NMI_max)
# print('ARI_max=',ARI_max)
# print('ACC_max=',ACC_max)
# =============================================================================
        

    
