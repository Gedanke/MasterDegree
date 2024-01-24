# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
from sklearn.datasets import *
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
import scipy.cluster.hierarchy as sch
from multiprocessing.pool import Pool
from .analyze import *

"""
绘图
"""


class PlotDemo:
    """
    根据 ./result/demo/analyze/ 下的数据集绘制图，将图存放到 ./result/demo/plot/
    """

    def __init__(self, path="./result/demo/analyze/", params={}) -> None:
        """
        初始化相关成员
        Args:
            path (str, optional): 文件路径，由于调用该类的主程序路径待定，这里使用手动传入，同时根据该路径解析出保存路径. Defaults to "./result/demo/analyze/".
            params (_type_, optional): 构造处理数据集需要的参数，这里不构造数据集，但参数中的规模要与 analyze 中的一致
            {
                "moons"/"circles...":
                {
                    "norm": 0/1, 归一化/标准化
                    "gmu": float, 高斯噪声中的 mu
                    "sigma": float, 高斯噪声中的 sigma
                    "type": moons/circles, 双月/双圈数据集
                    "num": int, 数据集样本个数,
                    "noise": float/list, 噪声级别,
                    "noise_type": 0/1, 高斯噪声/直接生成
                    "kmu": float, krod 的参数
                    "ckmu": float, ckrod 的参数
                    "factor": 双圈数据集的参数 factor
                }
            }
        """
        """文件路径"""
        self.path = path
        """构造处理数据集需要的参数"""
        self.params = params
        """保存结果路径"""
        self.save_path = self.path.replace("/demo/analyze", "/demo/plot", 1)

        """其他参数"""
        self.cicrles_title_list = [
            "Euclidean",
            "Manhattan",
            "Gaussian kernel",
            "ROD",
            "KROD",
            "CKROD",
        ]
        self.moons_title_list = [
            "Original data",
            "Noise data",
            "Euclidean",
            "Gaussian kernel",
            "Manhattan distance",
            "ROD",
            "KROD",
            "CKROD",
        ]

    def show_circles(self):
        """
        展示 demo 下的 circles 数据集结果
        """

    def show_moons(self):
        """
        展示 demo 下的 moons 数据集结果
        """
        """根据参数选择本次作图需要的数据集"""
        path_list = []
        print(
            os.listdir(self.path + "moons/noise_" + str(self.params["moons"]["noise"]))
        )

        return
        fig, axes = plt.subplots(1, 8, figsize=(16, 2))

        for i in range(8):
            if i < 2:
                """Original data，Noise data"""
                path = ""
            else:
                """其他数据集"""
                path = ""

            show_data = numpy.array(pandas.read_csv(path))
