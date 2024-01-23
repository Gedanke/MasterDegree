# -*- coding: utf-8 -*-


import numpy
from munkres import Munkres
from sklearn.metrics.cluster import *

"""
数据集的基本信息
"""
"""合成数据集"""
SYNTHESIS_PARAMS = {
    "aggregation": {
        "path": "./dataset/experiment/synthesis/aggregation/aggregation.csv",
        "save_path": "./result/synthesis/",
        "samples_num": 788,
        "features_num": 2,
        "num": 7,
    },
    "compound": {
        "path": "./dataset/experiment/synthesis/compound/compound.csv",
        "save_path": "./result/synthesis/",
        "samples_num": 399,
        "features_num": 2,
        "num": 6,
    },
    "D31": {
        "path": "./dataset/experiment/synthesis/D31/D31.csv",
        "save_path": "./result/synthesis/",
        "samples_num": 3100,
        "features_num": 2,
        "num": 31,
    },
    "flame": {
        "path": "./dataset/experiment/synthesis/flame/flame.csv",
        "save_path": "./result/synthesis/",
        "samples_num": 240,
        "features_num": 2,
        "num": 2,
    },
    "jain": {
        "path": "./dataset/experiment/synthesis/jain/jain.csv",
        "save_path": "./result/synthesis/",
        "samples_num": 373,
        "features_num": 2,
        "num": 2,
    },
    "pathbased": {
        "path": "./dataset/experiment/synthesis/pathbased/pathbased.csv",
        "save_path": "./result/synthesis/",
        "samples_num": 300,
        "features_num": 2,
        "num": 3,
    },
    "R15": {
        "path": "./dataset/experiment/synthesis/R15/R15.csv",
        "save_path": "./result/synthesis/",
        "samples_num": 600,
        "features_num": 2,
        "num": 15,
    },
    "S2": {
        "path": "./dataset/experiment/synthesis/S2/S2.csv",
        "save_path": "./result/synthesis/",
        "samples_num": 5000,
        "features_num": 2,
        "num": 15,
    },
    "spiral": {
        "path": "./dataset/experiment/synthesis/spiral/spiral.csv",
        "save_path": "./result/synthesis/",
        "samples_num": 312,
        "features_num": 2,
        "num": 3,
    },
}
"""uci 数据集"""
UCI_PATAMS = {
    "abalone": {
        "path": "./dataset/experiment/uci/abalone/abalone.csv",
        "save_path": "./result/uci/",
        "samples_num": 4177,
        "features_num": 8,
        "num": 3,
    },
    "blood": {
        "path": "./dataset/experiment/uci/blood/blood.csv",
        "save_path": "./result/uci/",
        "samples_num": 748,
        "features_num": 5,
        "num": 2,
    },
    "dermatology": {
        "path": "./dataset/experiment/uci/dermatology/dermatology.csv",
        "save_path": "./result/uci/",
        "samples_num": 366,
        "features_num": 34,
        "num": 6,
    },
    "ecoli": {
        "path": "./dataset/experiment/uci/ecoli/ecoli.csv",
        "save_path": "./result/uci/",
        "samples_num": 336,
        "features_num": 7,
        "num": 8,
    },
    "glass": {
        "path": "./dataset/experiment/uci/glass/glass.csv",
        "save_path": "./result/uci/",
        "samples_num": 214,
        "features_num": 9,
        "num": 6,
    },
    "iris": {
        "path": "./dataset/experiment/uci/iris/iris.csv",
        "save_path": "./result/uci/",
        "samples_num": 150,
        "features_num": 4,
        "num": 3,
    },
    "isolet": {
        "path": "./dataset/experiment/uci/isolet/isolet.csv",
        "save_path": "./result/uci/",
        "samples_num": 1560,
        "features_num": 617,
        "num": 26,
    },
    "jaffe": {
        "path": "./dataset/experiment/uci/jaffe/jaffe.csv",
        "save_path": "./result/uci/",
        "samples_num": 213,
        "features_num": 65536,
        "num": 10,
    },
    "letter": {
        "path": "./dataset/experiment/uci/letter/letter.csv",
        "save_path": "./result/uci/",
        "samples_num": 20000,
        "features_num": 16,
        "num": 26,
    },
    "libras": {
        "path": "./dataset/experiment/uci/libras/libras.csv",
        "save_path": "./result/uci/",
        "samples_num": 360,
        "features_num": 90,
        "num": 15,
    },
    "lung": {
        "path": "./dataset/experiment/uci/lung/lung.csv",
        "save_path": "./result/uci/",
        "samples_num": 203,
        "features_num": 3312,
        "num": 5,
    },
    "magic": {
        "path": "./dataset/experiment/uci/magic/magic.csv",
        "save_path": "./result/uci/",
        "samples_num": 19020,
        "features_num": 10,
        "num": 2,
    },
    "parkinsons": {
        "path": "./dataset/experiment/uci/parkinsons/parkinsons.csv",
        "save_path": "./result/uci/",
        "samples_num": 195,
        "features_num": 23,
        "num": 2,
    },
    "pima": {
        "path": "./dataset/experiment/uci/pima/pima.csv",
        "save_path": "./result/uci/",
        "samples_num": 768,
        "features_num": 8,
        "num": 2,
    },
    "seeds": {
        "path": "./dataset/experiment/uci/seeds/seeds.csv",
        "save_path": "./result/uci/",
        "samples_num": 210,
        "features_num": 7,
        "num": 3,
    },
    "segment": {
        "path": "./dataset/experiment/uci/segment/segment.csv",
        "save_path": "./result/uci/",
        "samples_num": 2310,
        "features_num": 19,
        "num": 7,
    },
    "sonar": {
        "path": "./dataset/experiment/uci/sonar/sonar.csv",
        "save_path": "./result/uci/",
        "samples_num": 208,
        "features_num": 60,
        "num": 2,
    },
    "spambase": {
        "path": "./dataset/experiment/uci/spambase/spambase.csv",
        "save_path": "./result/uci/",
        "samples_num": 4601,
        "features_num": 57,
        "num": 2,
    },
    "teaching": {
        "path": "./dataset/experiment/uci/teaching/teaching.csv",
        "save_path": "./result/uci/",
        "samples_num": 151,
        "features_num": 5,
        "num": 3,
    },
    "tox171": {
        "path": "./dataset/experiment/uci/tox171/tox171.csv",
        "save_path": "./result/uci/",
        "samples_num": 171,
        "features_num": 5748,
        "num": 4,
    },
    "twonorm": {
        "path": "./dataset/experiment/uci/twonorm/twonorm.csv",
        "save_path": "./result/uci/",
        "samples_num": 7400,
        "features_num": 20,
        "num": 2,
    },
    "usps": {
        "path": "./dataset/experiment/uci/usps/usps.csv",
        "save_path": "./result/uci/",
        "samples_num": 11000,
        "features_num": 256,
        "num": 10,
    },
    "waveform": {
        "path": "./dataset/experiment/uci/waveform/waveform.csv",
        "save_path": "./result/uci/",
        "samples_num": 5000,
        "features_num": 21,
        "num": 3,
    },
    "waveformNoise": {
        "path": "./dataset/experiment/uci/waveformNoise/waveformNoise.csv",
        "save_path": "./result/uci/",
        "samples_num": 5000,
        "features_num": 40,
        "num": 3,
    },
    "wdbc": {
        "path": "./dataset/experiment/uci/wdbc/wdbc.csv",
        "save_path": "./result/uci/",
        "samples_num": 569,
        "features_num": 30,
        "num": 2,
    },
    "wilt": {
        "path": "./dataset/experiment/uci/wilt/wilt.csv",
        "save_path": "./result/uci/",
        "samples_num": 4839,
        "features_num": 5,
        "num": 2,
    },
    "wine": {
        "path": "./dataset/experiment/uci/wine/wine.csv",
        "save_path": "./result/uci/",
        "samples_num": 178,
        "features_num": 13,
        "num": 3,
    },
}
"""图片数据集"""
IMAGE_PARAMS = {}
"""
一些公用的函数
"""


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
