import networkx as nx
import numpy as np


def extract_1hop_features(graph, edge_list):
    """
    Extract NNESF subgraph features for each edge in the graph.
    This function computes structural features like edge count and node count for specific subgraphs.

    :param graph: A NetworkX graph object (adjacency graph)
    :param edge_list: List of edges for which to compute features
    :return: A numpy array of extracted features
    """

    features = []
    for edge in edge_list:
        x, y = edge
        # Neighbors of x excluding y
        nei_x = set(graph.neighbors(x)) - {y}
        # Neighbors of y excluding x
        nei_y = set(graph.neighbors(y)) - {x}
        # Common neighbors of x and y
        cn_xy = set(nx.common_neighbors(graph, x, y))

        # Subgraphs for each group
        sub_x = graph.subgraph(nei_x)
        sub_y = graph.subgraph(nei_y)
        sub_cn = graph.subgraph(cn_xy)

        # Feature extraction
        fea = np.zeros(6)
        fea[0] = sub_cn.number_of_edges()  # Number of edges in common subgraph
        fea[1] = sub_cn.number_of_nodes()  # Number of nodes in common subgraph
        fea[2] = sub_x.number_of_edges()  # Number of edges in x's subgraph
        fea[3] = sub_x.number_of_nodes()  # Number of nodes in x's subgraph
        fea[4] = sub_y.number_of_edges()  # Number of edges in y's subgraph
        fea[5] = sub_y.number_of_nodes()  # Number of nodes in y's subgraph

        # Adding degree-based features (new features)
        fea = np.append(fea, [graph.degree[x], graph.degree[y]])

        features.append(fea)
    return np.array(features)


def extract_elementary_subgraph_features(graph, edge_list):
    """
    Extract elementary subgraph features for each edge in the graph.

    :param graph: A NetworkX graph object (adjacency graph)
    :param edge_list: List of edges for which to compute features
    :return: A numpy array of extracted features
    """
    features = []
    for edge in edge_list:
        x, y = edge
        # Common neighbors
        cn_xy = set(nx.common_neighbors(graph, x, y))

        # Elementary subgraph feature extraction
        fea = np.zeros(5)
        fea[0] = len(cn_xy)  # Number of common neighbors
        fea[1] = graph.degree[x] + graph.degree[y]  # Degree sum
        fea[2] = graph.subgraph(cn_xy).number_of_edges()  # Edges among common neighbors

        # Adding additional features for better distinction (new features)
        fea[3] = len(set(graph.neighbors(x)) & set(graph.neighbors(y)))  # Overlap of neighbors
        fea[4] = abs(graph.degree[x] - graph.degree[y])  # Degree difference

        features.append(fea)
    return np.array(features)
