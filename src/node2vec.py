# coding=utf-8
import numpy as np
import networkx as nx
import random
import logging


class Graph:
    def __init__(self, graphs, G, lam, is_directed, p, q):
        self.graphs = graphs
        self.G = G
        self.lam = lam
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
         从一个初始结点计算一个随机游走
        :param walk_length: 随机游走序列长度
        :param start_node: 初始结点
        :return: 列表，随机游走序列
        """
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]
        logging.info(str(start_node) + "random walk start...")
        while len(walk) < walk_length:
            cur = walk[-1]
            # 求当前结点的邻居结点
            cur_nbrs = sorted(G.neighbors(cur))
            # 如果存在邻居结点
            if len(cur_nbrs) > 0:
                # 如果序列中仅有一个结点，即第一次游走
                if len(walk) == 1:
                    """
                    结合cur_nbrs = sorted(G.neighbor(cur)) 和 alias_nodes/alias_edges的序号，
                    才能确定节点的ID。
                    所以路径上的每个节点在确定下一个节点是哪个的时候，都要经过sorted(G.neighbors(cur))这一步。
                    """
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                # 如果序列中有多个结点
                else:
                    # 找到当前游走结点的前一个结点
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
                                               alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break
        logging.info(str(start_node) + "random walk end...")
        return walk

    def simulate_walks(self, num_walks, walk_length):
        """
        Repeatedly simulate random walks from each node.
        对每个结点，根据num_walks得出其多条随机游走路径
        """
        logging.info("Repeatedly simulate random walks from each node...")
        G = self.G
        walks = []
        logging.info("all nodes to list")
        nodes = list(G.nodes())
        logging.info('Walk iteration:')
        for walk_iter in range(num_walks):
            logging.info(str(walk_iter + 1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
        logging.info("Walk iteration end")
        return walks

    def get_alias_edge(self, src, dst):
        """
        Get the alias edge setup lists for a given edge.
        :param src:  随机游走序列种的上一个结点
        :param dst:  当前结点
        :return:
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        # 这里可以进行优化，默认是选取所有的邻居结点
        # TODO：可以设置于一个阈值？
        # 三种情况
        for dst_nbr in sorted(G.neighbors(dst)):
            # 返回源结点
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
            # 源结点和这个目标结点的邻居结点之间有直连边
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            # 没有直连边
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
        norm_const = sum(unnormalized_probs)
        # 概率归一化
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        # 第一个返回值是Alias列表，第二个返回值是转移概率列表
        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self, temporary):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        logging.info("Start Preprocessing of transition probabilities for guiding the random walks.")
        if temporary:
            self.get_time_matrix()
        G = self.G
        is_directed = self.is_directed

        # 存储每个结点对应的两个采样列表
        alias_nodes = {}
        i = 0
        logging.info("nodes build start...")
        # G.nodes()返回一个结点列表
        for node in G.nodes():
            i = i + 1
            if i % 100000 == 0:
                logging.info(str(i) + " nodes have been build")
            # 得到当前结点的邻居结点(有直连关系)的权值列表，[1,1,1,1...]
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            # 权重求和
            norm_const = sum(unnormalized_probs)
            # 求每个权重的占的比重，权重大的占的比重就大
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)
        logging.info("nodes build end...")
        alias_edges = {}
        triads = {}
        logging.info("edges build start...")
        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            j = 0
            # G.edges()返回一个列表元组，列表里面是边关系，形如[(1,2), (1,3), ...]
            # (1,2)代表结点1和结点2之间有一条边
            for edge in G.edges():
                j = j + 1
                if j % 100000 == 0:
                    logging.info(str(j) + " alias_edges have been build")
                # 先构建(1,2)，再构建(2,1)
                # 这里复杂度较高，需要优化
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
        logging.info("edges build end...")
        # alias_nodes形式为{1:(J, q), 2:(J,q)...},1和2代表结点id
        # alias_edges形式为{(1,2):(J,q), (2,1):(J,q),(1,3):(J,q)...} (1,2)代表一条边
        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        logging.info("End --- Preprocessing of transition probabilities for guiding the random walks.")
        return

    def get_time_matrix(self):
        # 初始化权重，lam是每个时刻图对下一个时刻的影响参数，指数衰减
        lam = self.lam
        n = len(self.G.nodes())
        time_count = len(self.graphs)
        matrix_time = np.zeros((n, n))
        for g in self.graphs:
            for edge in g.edges():
                if g[edge[0]][edge[1]]['weight'] > 1:
                    g[edge[0]][edge[1]]['weight'] = 1
        # 计算权重
        for i, g in enumerate(self.graphs):
            t = np.array(nx.adjacency_matrix(g).todense())
            matrix_time = matrix_time + lam ** (time_count - i) * t
        rows, cols = matrix_time.shape
        # 直接通过邻接矩阵转化为图会丢失节点信息
        nodes = self.G.nodes()
        for i in range(rows):
            for j in range(cols):
                if matrix_time[i][j] > 0:
                    self.G.add_edge(nodes[i], nodes[j], weight=matrix_time[i][j])
        return matrix_time


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    alias_setup的作用是根据二阶random walk输出的概率变成每个节点对应两个数，被后面的alias_draw函数所进行抽样
    :param probs: 结点之间权重所占比例向量，是一个列表
    :return: 输入概率，得到对应的两个列表，
             一个是在原始的prob数组[0.4,0.8,0.6,1]，
             另外就是在上面补充的Alias数组，其值代表填充的那一列的序号索引
             具体的可以参见博客 https://blog.csdn.net/haolexiao/article/details/65157026
             方便后面的抽样调用
    """
    # J和q数组和probs数组大小一致
    # probs长度由当前结点的邻居节点数量决定
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)
    # 将数据分类为具有概率的结果 大于或者小于1 / K.
    # 这两个列表里存放的是结点的下标
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)
    # 然后循环并创建少量二元混合分布
    # 在整个均匀混合分布中适当地分配更大的结果。
    # 假如每条边权重都为1，实际上这里的while循环不会执行，因为每条边概率都是一样的，相当于不需要采样
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()  # smaller自己也会减少最右边的值
        large = larger.pop()
        # 在代码中的实现思路： 构建方法： 
        # 1.找出其中面积小于等于1的列，如i列，这些列说明其一定要被别的事件矩形填上，所以在Prab[i]中填上其面积 
        # 2.然后从面积大于1的列中，选出一个，比如j列，用它将第i列填满，然后Alias[i] = j，第j列面积减去填充用掉的面积。
        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q


def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    抽样函数
    使用alias采样从一个非均匀离散分布中采样
    :param J:
    :param q:
    :return:
    """
    K = len(J)

    # 从整体均匀混合分布中采样
    kk = int(np.floor(np.random.rand() * K))
    # 从二元混合中采样，要么保留较小的，要么选择更大的
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
