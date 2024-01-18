# -*- coding: utf-8 -*-


from codes.algorithm import *
from .dataSetting import *
from multiprocessing import Pool

conf = dict()
"""Demo 中使用到的度量方法，这里用简写"""
# MOONS_DIS_METHOD = ["euc", "man", "gau", "rod", "krod", "ckrod"]
MOONS_DIS_METHOD = ["euc", "man", "gau"]

"""
运行算法
"""


class RunDemo:
    """
    在 demo 数据集上进行实验
    使用一些度量方式，运行 demo 数据集(双月，双圈)，获得实验结果
    """

    def __init__(self, path, params={}) -> None:
        """
        初始化相关成员
        Args:
            path (_type_): 文件路径，由于调用该类的主程序路径待定，这里使用手动传入，同时根据该路径解析出保存路径
            params (_type_, optional): 构造处理数据集需要的参数，这里不构造数据集，但参数中的规模要与 experiment 中的一致
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
        self.save_path = self.path.replace("/experiment", "").replace(
            "dataset", "result"
        )
        """其他参数"""
        """不同类型数据的参数"""
        self.data_params = {}
        """样本集合，不含标签列"""
        self.samples = pandas.DataFrame({})
        """真实标签，默认数据的最后一列为标签"""
        self.label_true = list()

    def load_samples_msg(self, data_type, noise_level):
        """
        加载对应参数下的数据集
        Args:
            data_type (_type_): 不同类型的数据
            noise_level (_type_): 不同噪声级别
        """
        file_name = self.path + data_type + "/"
        if data_type == "moons":
            """双月数据集"""
            file_name += "moons__num_" + str(self.data_params["num"])
        elif data_type == "circles":
            """双圈数据集"""
            file_name += (
                "circles__num_"
                + str(self.data_params["num"])
                + "__factor_"
                + str(self.data_params["factor"])
            )

        """加上噪声水平"""
        file_name += "__noise_" + str(noise_level) + ".csv"
        """读取 csv 文件全部内容"""
        self.samples = pandas.read_csv(file_name)
        col = list(self.samples.columns)
        """self.samples，最后一列为标签列"""
        self.label_true = self.samples[col[-1]].tolist()
        """self.samples 其余列为数据列，不包括标签列"""
        self.samples = numpy.array(self.samples.loc[:, col[0:-1]])

    def deal_moons(self):
        """
        处理 moons 数据集，后期可以考虑使用多进程优化
        """
        """创建以该数据集命名的文件夹"""
        if not os.path.isdir(self.save_path + "result/moons"):
            os.mkdir(self.save_path + "result/moons")

        """加载 moons 数据集"""
        self.data_params = self.params["moons"]
        self.load_samples_msg("moons", self.data_params["noise"])

        """多进程"""
        p = Pool()
        """不同距离"""
        for dis_name in MOONS_DIS_METHOD:
            file_path = (
                self.save_path
                + "result/moons/"
                + "num_"
                + str(self.data_params["num"])
                + "__"
            )
            p.apply_async(
                multi_deal_demo,
                args=(
                    self.data_params,
                    self.samples,
                    dis_name,
                    file_path + dis_name + ".csv",
                ),
            )

        p.close()
        p.join()

    def deal_circles(self):
        """
        处理 circles 数据集
        """
        """创建以该数据集命名的文件夹"""
        if not os.path.isdir(self.save_path + "result/circles/"):
            os.mkdir(self.save_path + "result/circle/")

        """加载 moons 数据集"""
        self.data_params = self.params["circles"]
        """多进程"""
        p = Pool()
        """不同噪声级别"""
        for i in self.data_params["noise"]:
            self.load_samples_msg("circles", i)
            """不同距离"""
            """创建不同噪声级别的文件夹"""
            if not os.path.isdir(self.save_path + "result/circles/noise_" + str(i)):
                os.mkdir(self.save_path + "result/circles/noise_" + str(i))

            for dis_name in MOONS_DIS_METHOD:
                file_path = (
                    self.save_path
                    + "result/circles/noise_"
                    + str(i)
                    + "/num_"
                    + str(self.data_params["num"])
                    + "__"
                )
                p.apply_async(
                    multi_deal_demo,
                    args=(
                        self.data_params,
                        self.samples,
                        dis_name,
                        file_path + dis_name + ".csv",
                    ),
                )

        p.close()
        p.join()


class RunSynthesis:
    """
    在 synthesis 数据集上进行实验
    """


class RunUci:
    """
    在 uci 数据集上进行实验
    """


class RunImage:
    """
    在 image 数据集上进行实验
    """
