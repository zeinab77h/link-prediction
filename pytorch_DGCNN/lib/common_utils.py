# common_utils.py

import networkx as nx
import numpy as np

def extract_elementary_subgraph_features(graph, edge_list):
    """
    این تابع ویژگی‌های زیرگراف را استخراج می‌کند.
    """
    # پیاده‌سازی تابع
    features = []
    for edge in edge_list:
        x, y = edge
        nei_x = set(graph.neighbors(x)) - {y}
        nei_y = set(graph.neighbors(y)) - {x}
        cn_xy = set(nx.common_neighbors(graph, x, y))

        fea = np.zeros(6)
        fea[0] = len(cn_xy)  # تعداد همسایه‌های مشترک
        fea[1] = len(nei_x)
        fea[2] = len(nei_y)
        fea[3] = graph.degree(x)
        fea[4] = graph.degree(y)
        fea[5] = len(edge_list)
        features.append(fea)
    return np.array(features)
