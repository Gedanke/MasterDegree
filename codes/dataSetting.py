# -*- coding: utf-8 -*-


import math
import numpy
import pandas
from sklearn.datasets import *
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances


def noramlized(param, data):
    """
    数据归一化处理
    Args:
        param (_type_): 数据集的参数
        data (_type_): 原始数据，不包含标签
    Returns:
        new_data (_type_): 归一化后的数据
    """
    """归一化后的数据"""
    new_data = data.copy()

    if "norm" in param.keys():
        if param["norm"] == 0:
            """归一化"""
            new_data = (data - data.min()) / (data.max() - data.min())
        elif param["norm"] == 1:
            """标准化"""
            new_data = StandardScaler().fit_transform(data)
    else:
        """没有给就默认是归一化"""
        new_data = (data - data.min()) / (data.max() - data.min())

    return new_data


def normalized_label(label: list) -> list:
    """
    将标签转换为从 0 开始的整数，这里采用的方法是取最小的数，将所有的数据减去最小的数据
    Args:
        label (list): 原有标签
    Returns:
        new_label (list): 规整后的标签
    """
    """最小的数据"""
    min_value = min(label)
    """规整后的标签"""
    new_label = [int(value - min_value) for value in label]

    return new_label


def get_moons(params):
    """
    https://blog.csdn.net/chenxy_bwave/article/details/122078564
    生成并返回双月数据集
    Returns:
        data (_type_): 数据集
        label (_type_): 标签
    """
    if "noise" in params.keys():
        data, label = make_moons(
            n_samples=params["num"],
            shuffle=True,
            noise=params["noise"],
            random_state=1,
        )
    else:
        """无噪声"""
        data, label = make_moons(
            n_samples=params["num"],
            shuffle=True,
            noise=0,
            random_state=1,
        )

    return data, label


def get_circles(params):
    """
    https://blog.csdn.net/chenxy_bwave/article/details/122078564
    生成并返回双月数据集
    Returns:
        data (_type_): 数据集
        label (_type_): 标签
    """
    if "noise" in params.keys():
        data, label = make_circles(
            n_samples=params["num"],
            shuffle=True,
            noise=params["noise"],
            random_state=1,
            factor=params["factor"],
        )
    else:
        """无噪声"""
        data, label = make_circles(
            n_samples=params["num"],
            shuffle=True,
            noise=0,
            random_state=1,
            factor=params["factor"],
        )

    return data, label


def distance_gas(data_params, samples):
    """
    高斯度量
    Args:
        data_params (_type_): 数据集的参数
        samples (_type_): 数据集矩阵
    Returns:
        dis_matrix (_type_): 距离矩阵
    """
    """样本数量"""
    samples_num = data_params["num"]
    dis_matrix = numpy.zeros((samples_num, samples_num))

    for i in range(samples_num):
        for j in range(samples_num):
            dis_matrix[i, j] = math.exp(
                -sum((samples[i] - samples[j]) ** 2) / (0.2**2)
            )

    return dis_matrix


def distance_rods(data_params, samples, distance_method):
    """
    rod，krod，ckrod 度量
    Args:
        data_params (_type_): 数据集参数
        samples (_type_): 数据集矩阵
        distance_method (_type_): 不同度量方法
    Returns:
        dis_matrix (_type_): 距离矩阵
    """
    """样本数量"""
    samples_num = data_params["num"]
    """距离矩阵初始化"""
    dis_matrix = pandas.DataFrame(numpy.zeros((samples_num, samples_num)))
    """先根据欧式距离生成所有样本的距离列表"""
    dis_array = sch.distance.pdist(samples, "euclidean")
    """这里采用的方法是先深拷贝一份 self.dis_matrix"""
    euclidean_table = dis_matrix.copy()

    """处理距离矩阵"""
    num = 0
    for i in range(samples_num):
        for j in range(i + 1, samples_num):
            """赋值"""
            euclidean_table.at[i, j] = dis_array[num]
            """处理对角元素"""
            euclidean_table.at[j, i] = euclidean_table.at[i, j]
            num += 1

    """对 euclidean_table 使用 argsort()，该函数会对矩阵的每一行从小到大排序，返回的是 euclidean_table 中索引"""
    rank_order_table = numpy.array(euclidean_table).argsort()

    """对距离矩阵进行处理"""
    num = 0
    for i in range(samples_num):
        for j in range(i + 1, samples_num):
            """用新的度量方式得到的样本间距离覆盖掉 dis_array"""
            dis_array[num] = rods_fun(
                data_params, i, j, rank_order_table, euclidean_table, distance_method
            )
            """dis_matrix 内存放样本间的距离"""
            dis_matrix.at[i, j] = dis_array[num]
            """处理对角元素"""
            dis_matrix.at[j, i] = dis_matrix.at[i, j]
            num += 1

    return dis_matrix


def rods_fun(data_params, i, j, rank_order_table, euclidean_table, distance_method):
    """
    rod 及其改进算法
    Args:
        data_params (_type_): 数据集参数
        i (_type_): 第 i 个样本
        j (_type_): 第 j 个样本
        rank_order_table (_type_): 排序距离表
        euclidean_table (_type_): 欧式距离表
        distance_method (_type_): 度量方法
    Returns:
        dis (_type_): i 与 j 之间的距离
    """
    """样本数量"""
    samples_num = data_params["num"]
    """xi 与 xj 之间的距离"""
    dis = -1
    """第 i 个样本"""
    x1 = rank_order_table[i, :]
    """第 j 个样本"""
    x2 = rank_order_table[j, :]
    """rod 及其改进算法需要的一些前置条件"""
    """x1 样本的 id"""
    id_x1 = x1[0]
    """x2 样本的 id"""
    id_x2 = x2[0]
    """o_a_b，b 在 a 的排序列表中的索引位置"""
    o_a_b = numpy.where(x1 == id_x2)[0][0]
    """o_b_a，a 在 b 的排序列表中的索引位置"""
    o_b_a = numpy.where(x2 == id_x1)[0][0]

    """判断度量方法"""
    if distance_method == "rod":
        """d_a_b，在 a 的排序列表中，从 a 到 b 之间的所有元素在 b 的排序列表中的序数或者索引之和"""
        """先切片，a 的排序列表中 [a, b] 的索引列表"""
        slice_a_b = x1[0 : o_a_b + 1]
        """索引切片在 b 中的位置序数之和"""
        d_a_b = sum(numpy.where(x2 == slice_a_b[:, None])[-1])
        """d_b_a，在 b 的排序列表中，从 b 到 a 之间的所有元素在 a 的排序列表中的序数或者索引之和"""
        """先切片，b 的排序列表中 [b, a] 的索引列表"""
        slice_b_a = x2[0 : o_b_a + 1]
        """索引切片在 a 中的位置序数之和"""
        d_b_a = sum(numpy.where(x1 == slice_b_a[:, None])[-1])
        """rod"""
        dis = (d_a_b + d_b_a) / min(o_a_b, o_b_a)
    elif distance_method == "krod":
        """改进 rod"""
        l_a_b = o_a_b + o_b_a
        """高斯核"""
        k_a_b = math.exp(-((euclidean_table.at[i, j] / data_params["kmu"]) ** 2))
        """krod，改进 rod 与高斯核相结合"""
        dis = l_a_b / k_a_b
    elif distance_method == "ckrod":
        """改进 krod，数据都是二维的"""
        l_a_b = (o_a_b + o_b_a) / (samples_num - 1) / 2
        """高斯核"""
        k_a_b = math.exp(-((euclidean_table.at[i, j] / data_params["ckmu"]) ** 2))
        """ckrod，改进 krod 与高斯核相结合"""
        dis = l_a_b / k_a_b

    return dis


def multi_deal_demo(data_params, samples, dis_name, save_path):
    """
    多进程处理 moons 数据集
    Args:
        data_params (_type_): 数据集参数
        samples (_type_): 数据集矩阵
        dis_name (_type_): _description_
        save_path (_type_): _description_
    """
    """保存数据"""
    save_data = None
    """MDS 映射"""
    mds = MDS(dissimilarity="precomputed", random_state=0, normalized_stress="auto")

    """不同距离"""
    if dis_name == "euc":
        """欧式距离"""
        save_data = mds.fit_transform(euclidean_distances(samples))
    elif dis_name == "man":
        """曼哈顿距离"""
        save_data = mds.fit_transform(manhattan_distances(samples))
    elif dis_name == "gau":
        """高斯核"""
        save_data = mds.fit_transform(distance_gas(data_params, samples))
    elif dis_name in {"rod", "krod", "ckrod"}:
        "rod，krod，ckrod"
        save_data = mds.fit_transform(distance_rods(data_params, samples, dis_name))

    """归一化"""
    save_data = noramlized(save_data)
    """统一保存为 csv 文件"""
    pandas.DataFrame(save_data).to_csv(
        save_path,
        index=False,
    )
