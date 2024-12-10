import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from sklearn.metrics import accuracy_score
from munkres import Munkres


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="data/pubmed/", dataset="pubmed"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))#读取.content数据表

    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)#提取features并稀疏化表示

    labels = encode_onehot(idx_features_labels[:, -1])#one-hot编码

    #print(labels)
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)

    idx_map = {j: i for i, j in enumerate(idx)}

    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    G = nx.Graph()
    G.add_nodes_from(idx)
    G.add_edges_from(edges_unordered)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)#将原编号互作用对映射为现序号互作用对

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)#将非对称矩阵转化为对称矩阵
    #print(adj)
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    
# =============================================================================
#     idx_train = range(1000)
#     idx_val = range(1000,2000)
#     idx_test = range(2000,4000)
# =============================================================================
# =============================================================================
#     idx_train = range(100)
#     idx_val = range(200, 300)
#     idx_test = range(300, 500)
# =============================================================================

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    #print(labels)
# =============================================================================
#     idx_train = range(12)
#     idx_val = range(12)
#     idx_test = range(115)
#     idx_train = torch.LongTensor(idx_train)
#     idx_val = torch.LongTensor(idx_val)
#     idx_test = torch.LongTensor(idx_test)
# =============================================================================

    #print(labels)
    return adj, features, labels,idx, G


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy1(labels, preds): #其它对比算法使用该函数计算ACC指标
    preds = np.array(preds)
    labels = labels.numpy()
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

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
