# -*- coding:utf-8 -*-
'''
Reference

Paper
------------
Girvan M, Newman M E J. Community structure in social and biological networks[J]. Proceedings of the National Academy of
 the United States of America, 2002, 99(12): 7821-7826.
'''
import networkx as nx
import community as cm
import copy


def all_results(G):
    results = []
    m = G.number_of_edges()
    for i in range(20):
        print('大循环=',i)
        betweenness_dict = nx.edge_betweenness_centrality(G)
        edge_max_betweenness = max(betweenness_dict.items(), key=lambda x:x[1])[0]
        G.remove_edge(edge_max_betweenness[0], edge_max_betweenness[1])
        community = [list(subgraph) for subgraph in nx.connected_components(G)]
        community_dict = {node:0 for node in G.nodes()}
        for j in range(len(community)):
            print('小循环=',j)
            each = community[j]
            for node in each:
                community_dict[node] = j
        results.append(community_dict)
    return results

def GN_partition(G):
    G_copy = copy.deepcopy(G)
    results = all_results(G)
    modularities = [cm.modularity(results[i], G_copy) for i in range(len(results))]
    max_modularity = max(modularities)
    max_index = modularities.index(max_modularity)
    max_result = results[max_index]
    return list(max_result.values()), max_modularity

if __name__ == '__main__':

    filepath = r'../data/karate.gml'

    G = nx.read_gml(filepath)
    community, modularity = partition(G)
    print(community)
    print(modularity)





