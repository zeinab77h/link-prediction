import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
import pickle as cp
# import _pickle as cp  # python3 compatability
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
from sklearn import metrics
from gensim.models import Word2Vec
import warnings
from lib.common_utils import extract_elementary_subgraph_features


warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('%s/../../pytorch_DGCNN' % cur_dir)
import multiprocessing as mp
import torch
import torch.nn as nn  # Added for neural network modules

from torch_geometric.transforms import LineGraph
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.utils import from_networkx
from lib.subgraph_features import extract_1hop_features


class GNNGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree).values())

        if len(g.edges()) != 0:
            x, y = list(zip(*g.edges()))
            self.num_edges = len(x)
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])

        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):
            edge_features = nx.get_edge_attributes(g, 'features')
            assert (type(list(edge_features.values())[0]) == np.ndarray)
            edge_features = {(min(x, y), max(x, y)): z for (x, y), z in list(edge_features.items())}
            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])
            self.edge_features = np.concatenate(self.edge_features, 0)


# Updated to support subgraph features from both DiffPool and LGLP
class EnhancedGraph(GNNGraph):
    def __init__(self, g, label, node_tags=None, node_features=None):
        super().__init__(g, label, node_tags, node_features)
        self.subgraph_features = None


cmd_args = argparse.Namespace()
cmd_args.latent_dim = [32, 32, 32, 1]
cmd_args.hidden = 128
cmd_args.num_epochs = 15


class DiffPool(nn.Module):
    def __init__(self, input_dim, assign_dim):
        super(DiffPool, self).__init__()
        self.assign_dim = assign_dim
        self.input_dim = input_dim
        self.assign_matrix = nn.Linear(input_dim, assign_dim)
        self.gnn_layer = nn.Linear(assign_dim, assign_dim)

    def forward(self, x, edge_index, batch):
        assign_score = self.assign_matrix(x)
        x = self.gnn_layer(assign_score)
        batch_size = len(torch.unique(batch))

        pooled_x = torch.zeros(batch_size, self.assign_dim).to(x.device)
        for i in range(batch_size):
            pooled_x[i] = torch.mean(x[batch == i], dim=0)

        return pooled_x, edge_index, batch


def sample_neg(net, test_ratio=0.1, train_pos=None, test_pos=None, max_train_num=None):
    net_triu = ssp.triu(net, k=1)
    row, col, _ = ssp.find(net_triu)
    if train_pos is None or test_pos is None:
        perm = random.sample(list(range(len(row))), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    if max_train_num is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    train_num, test_num = len(train_pos[0]), len(test_pos[0])
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    while len(neg[0]) < train_num + test_num:
        i, j = random.randint(0, n - 1), random.randint(0, n - 1)
        if i < j and net[i, j] == 0:
            neg[0].append(i)
            neg[1].append(j)
    train_neg = (neg[0][:train_num], neg[1][:train_num])
    test_neg = (neg[0][train_num:], neg[1][train_num:])
    return train_pos, train_neg, test_pos, test_neg


def links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, h=1, max_nodes_per_hop=None, node_information=None):
    max_n_label = {'value': 0}

    def helper(A, links, g_label):
        start = time.time()
        pool = mp.Pool(mp.cpu_count())
        results = pool.map_async(parallel_worker, [((i, j), A, h, max_nodes_per_hop, node_information) for i, j in
                                                   zip(links[0], links[1])])
        remaining = results._number_left
        pbar = tqdm(total=remaining)
        while True:
            pbar.update(remaining - results._number_left)
            if results.ready(): break
            remaining = results._number_left
            time.sleep(1)
        results = results.get()
        pool.close()
        pbar.close()
        g_list = [EnhancedGraph(g, g_label, n_labels, n_features) for g, n_labels, n_features in results]
        max_n_label['value'] = max(max([max(n_labels) for _, n_labels, _ in results]), max_n_label['value'])
        end = time.time()
        print("Time eplased for subgraph extraction: {}s".format(end - start))
        return g_list

    print('Enclosing subgraph extraction begins...')
    train_graphs = helper(A, train_pos, 1) + helper(A, train_neg, 0)
    test_graphs = helper(A, test_pos, 1) + helper(A, test_neg, 0)
    print(max_n_label)
    return train_graphs, test_graphs, max_n_label['value']


def to_linegraphs(batch_graphs, max_n_label):
    graphs = []
    pbar = tqdm(batch_graphs, unit='iteration')
    for graph in pbar:
        edges = graph.edge_pairs
        edge_feas = edge_fea(graph, max_n_label) / 2
        edges, feas = to_undirect(edges, edge_fea)
        edges = torch.tensor(edges)
        data = Data(edge_index=edges, edge_attr=feas)
        data.num_nodes = graph.num_nodes
        data = LineGraph()(data)
        data['y'] = torch.tensor([graph.label])
        data.num_nodes = graph.num_edges

        if hasattr(graph, 'subgraph_features'):
            subgraph_features = graph.subgraph_features
            if subgraph_features is not None:
                data.x = torch.cat([data.x, torch.tensor(subgraph_features, dtype=torch.float)], dim=1)

        graphs.append(data)
    return graphs


def to_undirect(edges, edge_fea):
    """
    Converts directed edges to undirected edges by duplicating them.
    :param edges: Edge list as a NumPy array
    :param edge_fea: Edge features associated with the edges
    :return: Tuple of undirected edges and their features
    """
    edges = np.reshape(edges, (-1, 2))
    sr = np.array([edges[:, 0], edges[:, 1]], dtype=np.int64)
    fea_s = edge_fea[sr[0, :], :]
    fea_s = fea_s.repeat(2, 1)
    fea_r = edge_fea[sr[1, :], :]
    fea_r = fea_r.repeat(2, 1)
    fea_body = torch.cat([fea_s, fea_r], 1)
    rs = np.array([edges[:, 1], edges[:, 0]], dtype=np.int64)
    return np.concatenate([sr, rs], axis=1), fea_body


def parallel_worker(args):
    """
    Helper function for parallel processing of subgraph extraction and labeling.
    :param args: A tuple containing the required arguments for subgraph_extraction_labeling
    :return: Processed subgraph and associated features
    """
    return subgraph_extraction_labeling(*args)


def subgraph_extraction_labeling(ind, A, h=1, max_nodes_per_hop=None, node_information=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    nodes_dist = [0, 0]
    for dist in range(1, h + 1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes)
    subgraph = A[nodes, :][:, nodes]
    # apply node-labeling
    labels = node_label(subgraph)
    # get node features
    features = None
    if node_information is not None:
        features = node_information[nodes]
    # construct nx graph
    g = nx.from_scipy_sparse_array(subgraph)
    # remove link between target nodes
    if not g.has_edge(0, 1):
        g.add_edge(0, 1)
    return g, labels.tolist(), features


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res


def node_label(subgraph):
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0] + list(range(2, K)), :][:, [0] + list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels > 1e6] = 0  # set inf labels to 0
    labels[labels < -1e6] = 0  # set -inf labels to 0
    return labels


def edge_fea(graph, max_n_label):
    """
    Extracts edge features for the given graph and maximum node label.
    :param graph: Graph to extract edge features from
    :param max_n_label: Maximum node label to use for feature extraction
    :return: Tensor containing edge features
    """
    node_tag = torch.zeros(graph.num_nodes, max_n_label + 1)
    tags = graph.node_tags
    tags = torch.LongTensor(tags).view(-1, 1)
    node_tag.scatter_(1, tags, 1)
    return node_tag
