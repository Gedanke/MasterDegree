# -*- coding: utf-8 -*-


import numpy
from munkres import Munkres
from sklearn.metrics.cluster import *


def cluster_acc(label_true, label_pred):
    """
    使用了匈牙利算法的聚类准确度 ACC
    Args:
        label_true (_type_): 真实标签
        label_pred (_type_): 聚类标签
    """
    """将两个标签列表统一转换为 numpy 类型"""
    label_true = numpy.array(label_true)
    label_pred = numpy.array(label_pred)

    """真实标签去掉重复元素，由小到大排序"""
    label_true_label = numpy.unique(label_true)
    """标签大小"""
    label_true_num = len(label_true_label)
    """聚类标签去掉重复元素，由小到大排序"""
    label_pred_label = numpy.unique(label_pred)
    """标签大小"""
    label_pred_num = len(label_pred_label)
    """取最大值，由于此处的两个标签都是做过处理的，是一样大的"""
    num = numpy.maximum(label_true_num, label_pred_num)
    """"""
    matrix = numpy.zeros((num, num))

    for i in range(label_true_num):
        ind_cla_true = label_true == label_true_label[i]
        ind_cla_true = ind_cla_true.astype(float)
        for j in range(label_pred_num):
            ind_cla_pred = label_pred == label_pred_label[j]
            ind_cla_pred = ind_cla_pred.astype(float)
            matrix[i, j] = numpy.sum(ind_cla_true * ind_cla_pred)

    m = Munkres()
    index = m.compute(-matrix.T)
    index = numpy.array(index)

    c = index[:, 1]
    label_pred_new = numpy.zeros(label_pred.shape)
    for i in range(label_pred_num):
        label_pred_new[label_pred == label_pred_label[i]] = label_true_label[c[i]]

    """正确的结果数量"""
    right = numpy.sum(label_true[:] == label_pred_new[:])
    acc = right.astype(float) / (label_true.shape[0])

    return acc


def cluster_index(samples, label_true, label_pred, label_sign=True):
    """
    得出聚类结果，以字典形式储存
    Args:
        samples (_type_): 样本(不含真实标签)，应对外部评价指标
        label_true (_type_): 真实标签
        label_pred (_type_): 预测标签
        label_sign (bool, optional): 是否含有标签. Defaults to True.
    """
    """存放聚类结果的字典"""
    cluster_result = dict()
    """先计算外部评价指标，即不需要标签的"""
    """戴维森堡丁指数"""
    cluster_result["davies_bouldin"] = davies_bouldin_score(samples, label_pred)
    """CH 分数"""
    cluster_result["calinski_harabasz"] = calinski_harabasz_score(samples, label_pred)
    """轮廓系数"""
    cluster_result["silhouette_coefficient"] = silhouette_score(samples, label_pred)

    """考虑内部评价指标，需要标签的"""
    if label_sign:
        """聚类准确度 acc"""
        if len(set(label_true)) == len(set(label_pred)):
            cluster_result["cluster_acc"] = cluster_acc(label_true, label_pred)
        else:
            cluster_result["cluster_acc"] = -1

        """兰德指数"""
        cluster_result["rand_index"] = rand_score(label_true, label_pred)
        """调整兰德指数"""
        cluster_result["adjusted_rand_index"] = adjusted_rand_score(
            label_true, label_pred
        )
        """互信息"""
        cluster_result["mutual_info"] = mutual_info_score(label_true, label_pred)
        """标准化的互信息"""
        cluster_result["normalized_mutual_info"] = normalized_mutual_info_score(
            label_true, label_pred
        )
        """调整互信息"""
        cluster_result["adjusted_mutual_info"] = adjusted_mutual_info_score(
            label_true, label_pred
        )
        """同质性"""
        cluster_result["homogeneity"] = homogeneity_score(label_true, label_pred)
        """完整性"""
        cluster_result["completeness"] = completeness_score(label_true, label_pred)
        """调和平均"""
        cluster_result["v_measure"] = v_measure_score(label_true, label_pred)
        """融合了同质性、完整性、调和平均"""
        cluster_result["h_c_v_m"] = homogeneity_completeness_v_measure(
            label_true, label_pred
        )
        """Fowlkes-Mallows index"""
        cluster_result["fowlkes_mallows_index"] = fowlkes_mallows_score(
            label_true, label_pred
        )

    """对结果取 6 位有效数字"""
    for key, value in cluster_result.items():
        if type(value) != tuple:
            """非 h_c_v_m"""
            cluster_result[key] = round(value, 6)
        else:
            """h_c_v_m 指标"""
            cluster_result[key] = [round(v, 6) for v in value]

    return cluster_result
