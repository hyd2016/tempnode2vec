# coding=utf-8
"""
Reference implementation of node2vec.

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
"""

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import pandas as pd
import linkprediction
from scipy.io import loadmat
import time


def parse_args():
    """
    Parses the node2vec arguments.
    """
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='email-Eu-core.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=2,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def read_graph(slice_count):
    """
    Reads the input network in networkx.
    """
    col_index = ["x", "y", "time"]
    result = pd.read_table('email-Eu-core-temporal-Dept1.txt', sep=' ', header=None, names=col_index)
    list_G = []
    set_edge = set()
    max_time = result['time'].max()
    slice_num = max_time / slice_count + 1
    G_test = nx.Graph()
    G_normal = nx.Graph()
    # 添加所有节点到图中
    G_normal.add_nodes_from(result['x'].tolist())
    G_normal.add_nodes_from(result['y'].tolist())
    for i in range(1, slice_count + 1):
        G = nx.Graph()
        # 添加所有节点到图中
        G.add_nodes_from(result['x'].tolist())
        G.add_nodes_from(result['y'].tolist())
        # 获取某个时间切片所有节点对
        edge = result[(result['time'] >= (i - 1) * slice_num) & (result['time'] < i * slice_num)].iloc[:, 0:2]
        # 统计出现频率作为边权重
        weighted_edge = edge.groupby(['x', 'y']).size().reset_index()
        weighted_edge.rename(columns={0: 'frequency'}, inplace=True)
        weighted_edge_tuples = [tuple(xi) for xi in weighted_edge.values]
        G.add_weighted_edges_from(weighted_edge_tuples)
        # 测试集
        if i == slice_count:
            G_test = G
            for edge in G_test.edges():
                if G_test[edge[0]][edge[1]]['weight'] > 1:
                    G_test[edge[0]][edge[1]]['weight'] = 1
            continue
        weighted_edge['frequency'] = 1
        edge_tuples = [tuple(xi) for xi in weighted_edge.values]
        G_normal.add_weighted_edges_from(edge_tuples)
        list_G.append(G)

    return list_G, G_normal, G_test


def learn_embeddings(walks):
    """
    Learn embeddings by optimizing the Skipgram objective using SGD.
    """
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,
                     iter=args.iter)
    model.save_word2vec_format(args.output)
    return model


def main(args):
    """
    Pipeline for representational learning for all nodes in a graph.
    """
    temporary = True
    isEnron = False
    # temporary = False
    if isEnron:
        graphs, graph, graph_test = read_enron_graph()
    else:
        graphs, graph, graph_test = read_graph(20)
    print "读图时间", time.time()
    G = node2vec.Graph(graphs, graph, 0.8, args.directed, args.p, args.q)
    G.preprocess_transition_probs(temporary)
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    model = learn_embeddings(walks)
    predict = linkprediction.LinkPrediction(graph, graph_test, model)
    predict.predict()


def read_enron_graph():
    m = loadmat("enron.mat")
    data = m["enron"]["email_sender"][0, 0]
    data_size = data.shape
    list_graph = []
    nodes = list(range(14908))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for i in range(data_size[1]):
        G = nx.Graph()
        G.add_nodes_from(nodes)
        coo = data[0, i].tocoo(copy=False)
        edges = pd.DataFrame({'index': coo.row, 'col': coo.col, 'data': coo.data}
                             )[['index', 'col', 'data']].sort_values(['index', 'col']).reset_index(drop=True)
        edges_tuple = [tuple(edge) for edge in edges.values]
        graph.add_weighted_edges_from(edges_tuple)
        G.add_weighted_edges_from(edges_tuple)
        list_graph.append(G)
    return list_graph, graph, list_graph.pop(-1)


if __name__ == "__main__":
    args = parse_args()
    main(args)
