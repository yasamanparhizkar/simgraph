"""
Branched from my_simgraph_06.py
Only contains edge selection functions to use with Cheng's LMNN code. Edges between same-labeled nodes are allowed as well.

Author: Yasaman
Last modified: May 3, 2023
"""

import numpy as np
# import time
# import matplotlib.pyplot as plt


def get_edges_tt(lbls, D, cnstr_method, smpl_t, seed):
    """
    Return a set of edges in the training similarity graph. 
    Only different-labeled nodes can be connected to each other (since the GLR loss function does not care about same-labeled nodes).
    If Dt=None, all different-labeled nodes are connected. 
    Otherwise, Dt shows the maximum number of different-labeled nodes that each node is connected to. 
    The edges are chosen according to a strategy; either randomly, or based on the nodes' temporal closeness to each other.
    Since the graph is assumed to be undirected, each edge is only noted by one (i,j) pair.

    Input: lbls, D, cnstr_method, smpl_t, seed
    lbls - Nx1 vector of node labels (1 for 'spike'; -1 for 'no spike').
    D - maximum node degree in the similarity graph (default: None, all different-labeled nodes are connected)
    cnstr_method - method of choosing edges of the graph; either 'random' or 'time' which means edges are chosen based on the temporal closeness of their nodes (default: 'random') 
    smpl_t - Nx1 vector of time indices of nodes       
    seed - for random choice of edges (default: None, no new seed)

    Output: edges
    edges - list of (i,j) pairs representing edges. Node indices correspond with their order in 'lbls' and 'smpl_t'.    
    """

    if D is None:
        return get_edges_tt_full(lbls)
    elif cnstr_method == 'random':
        return get_edges_tt_maxdeg(lbls, D, seed)
    elif cnstr_method == 'time':
        return get_edges_tt_time(lbls, D, smpl_t)
    
def get_edges_tt_full(x):
    """
    Refer to documentation for 'get_edges_tt'
    """

    edges = []
    N = x.shape[0]
    for i in range(N):
        for j in range(i+1, N):
                edges.append((i,j))
    return edges

def get_edges_tt_maxdeg(x, D, seed):
    """
    Refer to documentation for 'get_edges_tt'
    """

    if seed is not None:
        np.random.seed(seed)

    edges = []
    N = x.shape[0]
    degree = np.zeros(N)
    for i in range(N):
        candids = []
        for j in range(i+1, N):
            if degree[j] < D:
                candids.append(j)
        
        if (D-degree[i]) < len(candids):
            temp = np.random.choice(candids, size=int(D-degree[i]), replace=False)
        else:
            temp = np.array(candids)

        temp = [(i,j) for j in temp]   
        edges = edges + temp
        degree[i] += len(temp)
        for (i,j) in temp:
            degree[j] += 1

    return edges

def get_edges_tt_maxdeg(x, D, seed):
    """
    Refer to documentation for 'get_edges_tt'
    """

    if seed is not None:
        np.random.seed(seed)

    edges = []
    N = x.shape[0]
    degree = np.zeros(N)
    for i in range(N):
        candids = []
        for j in range(i+1, N):
            if degree[j] < D:
                candids.append(j)
        
        if (D-degree[i]) < len(candids):
            temp = np.random.choice(candids, size=int(D-degree[i]), replace=False)
        else:
            temp = np.array(candids)

        temp = [(i,j) for j in temp]   
        edges = edges + temp
        degree[i] += len(temp)
        for (i,j) in temp:
            degree[j] += 1

    return edges

def get_edges_tt_time(x, D, smpl_t):
    """
    Refer to documentation for 'get_edges_tt'
    """

    edges = []
    N = x.shape[0]
    degree = np.zeros(N)
    for i in range(N):
        candids = []
        for j in range(i+1, N):
            if degree[j] < D:
                candids.append(j)
        
        if (D-degree[i]) < len(candids):
            candids = np.array(sorted(candids, key=lambda l: np.abs(smpl_t[l]-smpl_t[i])))
            temp = candids[:int(D-degree[i])]
        else:
            temp = np.array(candids)

        temp = [(i,j) for j in temp]   
        edges = edges + temp
        degree[i] += len(temp)
        for (i,j) in temp:
            degree[j] += 1

    return edges


def get_edges_vv(num_val, graph_params, seed):
    """
    Return a set of edges between validation nodes in the similarity graph model.
    If cnstr_method is 'random' then edges are chosen randomly and Dv defines the maximum possible node degree.
    If cnstr_method is 'time' then edges are chosen based on the temporal closeness of their nodes; the closer the better.
    
    Inputs: num_val, graph_params, seed
    num_val - number of validation nodes
    graph_params - dict of parameters for graph construction and the penalty term:
        'Dv': maximum node degree in the validation graph (default: None, all different-labeled nodes are connected)
        'cnstr_method_vv': method of choosing edges of the validation graph; either 'random' or 'time' which means edges are chosen based on the temporal closeness of their nodes (default: 'random')
        'val_t': Nvx1 vector of time indices of validation nodes
    seed - for random choice of edges (default: None, no new seed)
    
    Outputs: edges_vv
    edges_vv - list of (i, j) pairs representing edges. Node indices conform to their order in val_t.
    """
    # unpack params
    Dv = graph_params['Dv'] if 'Dv' in graph_params else None
    cnstr_method_vv = graph_params['cnstr_method_vv'] if 'cnstr_method_vv' in graph_params else 'random'
    val_t = graph_params['val_t'] if 'val_t' in graph_params else None
    
    x = np.arange(num_val)
    return get_edges_tt(x, Dv, cnstr_method_vv, val_t, seed)

def get_edges_vt(num_val, train_y, graph_params, seed):
    """
    Return a set of edges between validation and training nodes. 
    Training nodes are assumed to belong to two clusters: -1 and 1.
    Each validation node is connected to exactly Dvt nodes of each cluster, 
    so there are a total of 2*Dvt edges connecting each validation node to the training nodes.
    Training nodes to be connected to each validation node are chosen either randomly or based on their temporal closeness to the validation node.
    
    Inputs: num_val, train_y, graph_params, seed
    num_val - number of validation nodes
    train_y - training nodes' labels used for clustering (-1 for 'no spike', 1 for 'spike')
    graph_params - dict of parameters for graph construction and the penalty term:
        'Dvt': number of edges between each validation node and each cluster of training nodes (default: None, each validation node is connected to all training nodes)
        'cnstr_method_vt': same as above, but for the edges between training and validation nodes
        'train_t': Ntx1 vector of time indices of training nodes
        'val_t': Nvx1 vector of time indices of validation nodes
    seed - for random selection of training nodes (default: None, no new seed)
    
    Outputs: edges_vt
    edges_vt - list of (i, j) pairs indicating edges between validation and training nodes. 
               first index refers to validation nodes and second index to training nodes.
               first index corresponds to the order of nodes in val_t.
               second index corresponds to the order of nodes in train_y and train_t.
    """
    # unpack params
    Dvt = graph_params['Dvt'] if 'Dvt' in graph_params else None
    train_t = graph_params['train_t'] if 'train_t' in graph_params else None
    val_t = graph_params['val_t'] if 'val_t' in graph_params else None

    if Dvt is None:
        return get_edges_vt_full(len(train_y), num_val)
    elif graph_params['cnstr_method_vt'] == 'random':
        return get_edges_vt_maxdeg(train_y, num_val, Dvt, seed)
    elif graph_params['cnstr_method_vt'] == 'time':
        return get_edges_vt_time(train_y, num_val, Dvt, train_t, val_t)
    
def get_edges_vt_full(num_train, num_val):
    """
    Refer to documentation for 'get_edges_vt'
    """
    
    edges_vt = []
    for i in range(num_val):
        for j in range(num_train):
            edges_vt.append((i, j))
            
    return edges_vt
    
def get_edges_vt_maxdeg(train_lbls, num_val, Dvt, seed):
    """
    Refer to documentation for 'get_edges_vt'
    """
    
    # create lists of indices of training nodes in each cluster (-1 and 1)
    c_neg = []
    c_pos = []
    for i, j in enumerate(train_lbls):
        if j == -1:
            c_neg.append(i)
        else:
            c_pos.append(i)
    
    # the requested number of edges can not be bigger than the minimum cluster size
    max_Dvt = min(len(c_neg), len(c_pos))
    if Dvt > max_Dvt:
        Dvt = max_Dvt
    
    # choose edges randomly
    rng = np.random.default_rng(seed)
    edges_vt = []   
    for v_node in range(num_val):
        edges_vt += [(v_node, j) for j in rng.choice(c_neg, size=Dvt, replace=False)]
        edges_vt += [(v_node, j) for j in rng.choice(c_pos, size=Dvt, replace=False)]
    
    return edges_vt

def get_edges_vt_time(train_lbls, num_val, Dvt, train_t, val_t):
    """
    Refer to documentation for 'get_edges_vt'
    """
    
    # create lists of indices of training nodes in each cluster (-1 and 1)
    c_neg = []
    c_pos = []
    for i, j in enumerate(train_lbls):
        if j == -1:
            c_neg.append(i)
        else:
            c_pos.append(i)
    
    # the requested number of edges can not be bigger than the minimum cluster size
    max_Dvt = min(len(c_neg), len(c_pos))
    if Dvt > max_Dvt:
        Dvt = max_Dvt
    
    # choose edges randomly
    edges_vt = []   
    for v_node in range(num_val):
        edges_vt += [(v_node, j) for j in sorted(c_neg, key= lambda l: np.abs(train_t[l]-val_t[v_node]))[:Dvt]]
        edges_vt += [(v_node, j) for j in sorted(c_pos, key= lambda l: np.abs(train_t[l]-val_t[v_node]))[:Dvt]]
    
    return edges_vt
