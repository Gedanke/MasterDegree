# -*- coding: utf-8 -*-


import copy
import os
import shutil
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
from .experiment import *


"""
对算法在数据集上获得的结果进行分析，这里不做图
"""


class AnalyzeDemo:
    """
    分析 demo 数据集，目前来看，暂时不需要分析，将特定参数产生的数据集结果复制到 ./result/demo/analyze 下，后续直接根据结果绘图即可
    """

    def __init__(self, path="./result/demo/result/", params={}) -> None:
        """
        初始化相关成员
        Args:
            path (str, optional): 文件路径，由于调用该类的主程序路径待定，这里使用手动传入，同时根据该路径解析出保存路径. Defaults to "./result/demo/result/".
            params (_type_, optional): 构造处理数据集需要的参数，这里不构造数据集，但参数中的规模要与 result 中的一致
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
        self.save_path = self.path.replace("/demo/result", "/demo/analyze", 1)

    def analyze_demo(self):
        """
        移动文件
        """
        for _dir in os.listdir(self.path):
            """尝试创建文件夹"""
            if not os.path.isdir(self.save_path + _dir):
                os.mkdir(self.save_path + _dir)
            for __dir in os.listdir(self.path + _dir):
                """文件夹，尝试创建文件夹"""
                if not os.path.isdir(self.save_path + _dir + "/" + __dir):
                    os.makedirs(self.save_path + _dir + "/" + __dir)
                for __dir_file in os.listdir(self.path + _dir + "/" + __dir):
                    """csv 文件，根据参数，移动对应的文件"""
                    if (
                        __dir_file.find(".csv") != -1
                        and __dir_file.find(str(self.params[_dir]["num"])) != -1
                    ):
                        shutil.copyfile(
                            self.path + _dir + "/" + __dir + "/" + __dir_file,
                            self.save_path + _dir + "/" + __dir + "/" + __dir_file,
                        )

        """在 mnoons 中，将原始数据集，与噪声数据集移动到 self.save_path + moons 下"""
        data_path = self.path.replace("/result/demo/result/", "/dataset/")
        """原始数据集"""
        shutil.copy(
            data_path
            + "data/demo/moons__num_"
            + str(self.params["moons"]["num"])
            + ".csv",
            self.save_path + "/moons/num_" + str(self.params["moons"]["num"]) + ".csv",
        )
        """噪声数据集"""
        shutil.copy(
            data_path
            + "experiment/demo/moons/moons__num_"
            + str(self.params["moons"]["num"])
            + "__noise_0.15.csv",
            self.save_path
            + "/moons/num_"
            + str(self.params["moons"]["num"])
            + "__noise"
            + ".csv",
        )

        """在 circless 中，将原始数据集，与噪声数据集移动到 self.save_path + circless 下以及不同的 noise_level 文件夹下"""
        """原始数据集"""
        shutil.copy(
            data_path
            + "data/demo/circles__num_"
            + str(self.params["circles"]["num"])
            + "__factor_"
            + str(self.params["circles"]["factor"])
            + ".csv",
            self.save_path
            + "/circles/num_"
            + str(self.params["circles"]["num"])
            + ".csv",
        )
        """噪声数据集"""
        for i in self.params["circles"]["noise"]:
            shutil.copy(
                data_path
                + "experiment/demo/circles/circles__num_"
                + str(self.params["circles"]["num"])
                + "__factor_"
                + str(self.params["circles"]["factor"])
                + "__noise_"
                + str(i)
                + ".csv",
                self.save_path
                + "/circles/noise_"
                + str(i)
                + "/num_"
                + str(self.params["circles"]["num"])
                + ".csv",
            )


class AnalyzeSynthesis:
    """
    分析 synthesis 数据集，根据数据集的实验结果选择最佳的结果，将文件保存到 ./result/
    """


class AnalyzeUci:
    """
    分析 UCI 数据集，根据数据集的实验结果选择最佳的结果，将文件保存到 ./result/
    """


class AnalyzeImage:
    """
    分析 image 数据集，根据数据集的实验结果选择最佳的结果，将文件保存到 ./result/
    """
