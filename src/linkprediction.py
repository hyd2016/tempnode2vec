# coding=utf-8
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
import time


class LinkPrediction:
    def __init__(self, graph_train, graph_test, model):
        self.graph_train = graph_train
        self.graph_test = graph_test
        self.model = model

    def data_label(self, graph):
        """
        从图中和model计算边的类别和边的特征，如果便存在，则为1，不存在为0，特征为节点特征的积edge(u,v) = f(u)*f(v）
        :param graph:
        :return:
        """
        model = self.model
        label = []
        features = []
        size_nodes = len(graph.nodes())
        # 遍历图的所有可能的边
        for i in range(size_nodes):
            for j in range(i + 1, size_nodes):
                feature = model[str(graph.nodes()[i])] * model[str(graph.nodes()[j])]
                if graph.has_edge(i, j):
                    label.append(1)
                else:
                    label.append(0)
                features.append(feature)
        return features, label

    def predict(self):
        """
        通过SVM进行分类，并进行预测
        :return:
        """
        svm_rbf = svm.SVC(C=0.8, gamma=200, class_weight='balanced')
        # svm_rbf = LogisticRegression(class_weight='balanced')
        train_feature, train_label = self.data_label(self.graph_train)
        train_feature = np.array(train_feature)
        # te = max(train_feature)
        tt = np.max(train_feature)
        train_feature = train_feature/np.max(train_feature)
        svm_rbf.fit(train_feature, train_label)
        true_features, true_label = self.data_label(self.graph_test)
        true_features = np.array(true_features)
        true_features = true_features/np.max(true_features)
        predict_probability = svm_rbf.predict(true_features)
        # predict_probability = svm_rbf.predict_proba(true_features)[:, 1]
        # fpr, tpr, thresholds = metrics.roc_curve(true_label, predict_probability, pos_label=1)
        # plt.plot(fpr, tpr, marker='o')
        # plt.show()
        auc_score = roc_auc_score(true_label, predict_probability)
        print auc_score

