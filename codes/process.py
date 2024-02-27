# -*- coding: utf-8 -*-


import os
import random
import numpy
import pandas
from scipy.io import loadmat
from .dataSetting import *

"""保留三位有效数字"""
numpy.set_printoptions(precision=3)
DATA_PATH = "./dataset/"


class DealData:
    """
    数据处理类，提供不同的方法处理实验所需的不同类型数据(位于 raw 下)
    所有的数据集格式都是 pandas.DataFrame，列名从 0 开始，最后一列 label 为标签，精度保留2位数即可，不需要高精度
    除了图像数据集外，实验所需的数据都需要归一化
    """

    def __init__(self, path, params={}):
        """
        初始化相关成员
        Args:
            path (_type_): demo/synthesis/uci/image，要处理的数据路径
            params (_type_, optional): 构造处理数据集需要的参数
            {
                "norm": 0/1, 归一化/标准化
                "gmu": float, 高斯噪声中的 mu
                "sigma": float, 高斯噪声中的 sigma
                "type": moons/circles, 双月/双圈数据集
                "num": int, 数据集样本个数,
                "noise": float, 噪声级别,
                "noise_type": 0/1, 高斯噪声/直接生成
                "mu": float, krod,ckrod 中的高斯核 mu
                "factor": 双圈数据集的参数 factor
            }
            使用前传入正确的参数，类中不对参数进行校验
        """
        """要处理的数据路径"""
        self.path = path
        """构造处理数据集需要的参数"""
        self.params = params
        """文件名"""
        self.file_name = ""

    def add_noise(self, data):
        """
        往数据中添加高斯噪声
        Args:
            data (_type_): 原始数据
        Returns:
            _type_: 添加高斯噪声后的数据
        """
        """深拷贝一份"""
        noise_data = data.copy()
        """样本数量"""
        num = len(noise_data)
        """列名"""
        col = list(noise_data.columns)

        """添加高斯噪声"""
        for i in range(num):
            for j in col:
                noise_data.at[i, j] += random.gauss(
                    self.params["gmu"], self.params["sigma"]
                )

        return noise_data

    def deal_demo(self):
        """
        https://blog.csdn.net/chenxy_bwave/article/details/122078564
        这里的数据是不含噪声的
        处理 demo 的数据集，原始数据存同时放在 raw 的 demo 文件夹下，归一化后的数据存同时放在 data 的 demo 文件夹下
        """
        """文件名"""
        self.file_name = "num_" + str(self.params["num"])
        """均无噪声，只有 demo 这块需要指定数据集类型 type，其他三个部分均不需要"""
        if self.params["type"] == "moons":
            self.file_name = "moons__" + self.file_name
            data, label = get_moons({"num": self.params["num"]})
        elif self.params["type"] == "circles":
            self.file_name = (
                "circles__" + self.file_name + "__factor_" + str(self.params["factor"])
            )
            data, label = get_circles(
                {"num": self.params["num"], "factor": self.params["factor"]}
            )

        """数据列保留两位有效数字"""
        data = data.round(3)
        """修改标签列名"""
        label = pandas.DataFrame(label)
        label.columns = ["label"]

        """保留原始数据到 raw 的 demo 下，合并 data 与 label"""
        pandas.DataFrame(data).join(pandas.DataFrame(label)).to_csv(
            DATA_PATH + "raw/" + self.path + "/" + self.file_name + ".csv", index=False
        )

        """将结果归一化，列名规整后保存到 data 的 demo 文件夹下"""
        data_data = pandas.DataFrame(noramlized(self.params, data)).round(3)
        data_data.columns = list(range(int(data.shape[1])))
        data_data.join(pandas.DataFrame(label)).to_csv(
            DATA_PATH + "data/" + self.path + "/" + self.file_name + ".csv", index=False
        )

    def deal_synthesis(self):
        """
        处理 raw 下的 synthesis 数据集，将处理好的原始数据集存放到 data 下的 synthesis 文件下
        """
        """raw 下的 synthesis 数据集均为 mat 文件"""
        dir_path = DATA_PATH + "raw/" + self.path + "/"

        for file in os.listdir(dir_path):
            """加载数据集"""
            raw_data = pandas.read_csv(dir_path + file, sep="\t")
            col_list = list(raw_data.columns)
            """标签规整，数据部分归一化"""
            if len(col_list) > 2:
                """大于两列，默认最后一列是标签"""
                label = {"label": normalized_label(list(raw_data[col_list[-1]]))}
                data = (
                    noramlized(self.params, raw_data[col_list[0:-1]])
                    .round(3)
                    .join(pandas.DataFrame(label))
                )
                """修改列标签"""
                data.columns = list(range(len(data.columns) - 1)) + ["label"]
            else:
                """小于等于列，此处默认没有列标签"""
                data = noramlized(self.params, raw_data).round(3)
                """修改列标签"""
                data.columns = list(range(len(data.columns)))

            data.to_csv(
                DATA_PATH
                + "data/"
                + self.path
                + "/"
                + os.path.splitext(file)[0]
                + ".csv",
                index=False,
            )

    def deal_uci(self):
        """
        处理 raw 下的 uci 数据集，将处理好的原始数据集存放到 data 下的 uci 文件下
        raw 下的 uci 数据集都有单独的文件夹，如果一个数据集的名称为 dataset，数字越小，表示有衔接越高
        dataset.data 原始未经处理的数据集，标签统一放在最后一列，分隔符统一为 ,(排除仅含 dataset.mat 外，原始数据集一定存在)
        dataset.names 对该数据集的详细介绍(大体上与 data 一同出现)
        dataset.txt 对该数据集的简略介绍(大体上与 data 一同出现)
        dataset.mat mat 格式的数据，可以直接读取，但不保证 key 是相同(一定存在该数据集)
        minmax_dataset.mat 经过归一化后的数据集(大部分存在)
        """
        dir_path = DATA_PATH + "raw/" + self.path + "/"

        for dataset in os.listdir(dir_path):
            """优先加载 minmax_dataset.mat，其次 dataset.mat，最后 dataset.data"""
            file_list = os.listdir(dir_path + dataset)
            file = "minmax_" + dataset + ".mat"
            """数据与标签"""
            raw_data = pandas.DataFrame({})
            label = list()
            if file in file_list:
                """读取 minmax_dataset.mat，都有 minmax_scaling 这个 key"""
                mat_data = loadmat(dir_path + dataset + "/" + file)
                raw_data = noramlized(
                    self.params, pandas.DataFrame(mat_data["minmax_scaling"][:, 1:])
                )
                label = normalized_label(list(mat_data["minmax_scaling"][:, 0]))
            else:
                """下面的处理条件适用于没有 minmax_dataset.mat 的情况(有 minmax_dataset.mat 的情况下，下面的判断条件可能不适用)，处理 dataset.mat 以及 dataset.data"""
                file = dataset + ".mat"
                if file in file_list:
                    """处理 dataset.mat，key 不统一，分别判断下即可，目前而言 X 是数据，Y 是标签"""
                    mat_data = loadmat(dir_path + dataset + "/" + file)
                    raw_data = noramlized(self.params, pandas.DataFrame(mat_data["X"]))
                    label = normalized_label([_[0] for _ in mat_data["Y"].tolist()])
                else:
                    """处理 dataset.data，现阶段不会进行到这一步，mat 格式文件是一定有的"""
                    pass
            """合并数据集"""
            data = raw_data.round(3).join(pandas.DataFrame({"label": label}))
            """以防万一，修改下列标签，这里暂时不创建文件夹"""
            data.columns = list(range(len(data.columns) - 1)) + ["label"]
            data.to_csv(
                DATA_PATH + "data/" + self.path + "/" + dataset + ".csv",
                index=False,
            )

    def deal_image(self):
        """
        处理 raw 下的 image 数据集，将处理好的原始数据集存放到 data 下的 image 文件下
        图像数据集冗余度高，可以使用 csv 格式，也可以把 mat 文件移动过去
        """
        dir_path = DATA_PATH + "raw/" + self.path + "/"

        for dataset in os.listdir(dir_path):
            """加载数据"""
            data = loadmat(dir_path + dataset + "/" + dataset + ".mat")
            # dd = data["X"][0:101]

            # fig, axes = plt.subplots(10, 10, figsize=(18, 18))
            # for i in range(10):
            #     for j in range(10):
            #         d = dd[i * 10 + j].reshape(
            #             int(math.sqrt(int(dd[i * 10 + j].shape[0]))),
            #             int(math.sqrt(int(dd[i * 10 + j].shape[0]))),
            #         )
            #         axes[i][j].imshow(d)
            #         axes[i][j].set_xticks(())
            #         axes[i][j].set_yticks(())

            # plt.tight_layout()
            # plt.subplots_adjust(wspace=0, hspace=0)
            # plt.show()

    def get_demo(self):
        """
        处理 data 路径下的数据集，将实验所需的数据集存放到 experiment 下的 demo 文件下
        这个类型的数据在添加噪声时候，有两种情况
        可以直接在参数中指定噪声级别，重新使用 get_type 方法获取新的数据集 => "noise_type": 0
        可以在 data 下的 demo 数据中添加高噪声 => "noise_type": 1，默认第一种
        """
        """判断是否有 noise_type 参数"""
        if "noise_type" not in self.params.keys():
            self.params["noise_type"] = 1
        """判断是否有 noise 参数"""
        if "noise" not in self.params.keys():
            self.params["noise"] = 0

        """
        判断 self.file_name 是否为空
        不为空，按照正常的流程，是先调用 deal_demo，根据参数创建对应的数据集，如果接下来继续调用 get_deom，那么 file_name 就不为空，我们就处理该文件
        为空，则说明是一次独立的调用，之前没有调用过 deal_demo，则我们读取 data 下所有的数据进行统一处理
        """
        file_name_list = []
        if self.file_name == "":
            """获取文件名"""
            file_name_list = [
                os.path.splitext(file)[0]
                for file in os.listdir(DATA_PATH + "data/" + self.path + "/")
            ]
        else:
            file_name_list.append(self.file_name)

        """两种不同的噪声添加方式"""
        if self.params["noise_type"] == 0:
            """读取文件名，解析参数，生成对应的噪声数据集"""
            for file in file_name_list:
                """这里无需判断是否有 noise，因为 data 下是没有噪声的"""
                param_list = file.split("__")
                data, label = None, None
                """判断数据类型"""
                if param_list[0] == "moons":
                    """moons，解析剩余参数"""
                    param = dict()
                    for ele in param_list[1:]:
                        if "num" in ele:
                            param[ele.split("_")[0]] = int(ele.split("_")[1])
                        else:
                            param[ele.split("_")[0]] = float(ele.split("_")[1])
                    param["noise"] = self.params["noise"]
                    data, label = get_moons(param)
                elif param_list[0] == "circles":
                    """circles，解析剩余参数"""
                    param = dict()
                    for ele in param_list[1:]:
                        if "num" in ele:
                            param[ele.split("_")[0]] = int(ele.split("_")[1])
                        else:
                            param[ele.split("_")[0]] = float(ele.split("_")[1])
                    param["noise"] = self.params["noise"]
                    data, label = get_circles(param)

                """创建以数据集命名的文件夹"""
                if not os.path.isdir(
                    DATA_PATH + "experiment/" + self.path + "/" + param_list[0]
                ):
                    os.mkdir(
                        DATA_PATH + "experiment/" + self.path + "/" + param_list[0]
                    )

                """归一化数据并保存结果"""
                data = pandas.DataFrame(noramlized(self.params, data).round(3))
                """修改标签列名"""
                data.columns = list(range(int(data.shape[1])))
                label = pandas.DataFrame(label)
                label.columns = ["label"]
                data.join(pandas.DataFrame(label)).to_csv(
                    DATA_PATH
                    + "experiment/"
                    + self.path
                    + "/"
                    + param_list[0]
                    + "/"
                    + file
                    + "__noise_"
                    + str(self.params["noise"])
                    + ".csv",
                    index=False,
                )

        elif self.params["noise_type"] == 1:
            """读取数据，添加高斯噪声，最后一列不要加入"""
            for file in file_name_list:
                data = pandas.read_csv(
                    DATA_PATH + "data/" + self.path + "/" + file + ".csv"
                )
                col_list = list(data.columns)
                param_list = file.split("__")

                """创建以数据集命名的文件夹"""
                if not os.path.isdir(
                    DATA_PATH + "experiment/" + self.path + "/" + param_list[0]
                ):
                    os.mkdir(
                        DATA_PATH + "experiment/" + self.path + "/" + param_list[0]
                    )

                """归一化"""
                noise_data = (
                    noramlized(self.params, self.add_noise(data[col_list[0:-1]]))
                    .round(3)
                    .join(data[col_list[-1]])
                )
                noise_data.to_csv(
                    DATA_PATH
                    + "experiment/"
                    + self.path
                    + "/"
                    + param_list[0]
                    + "/"
                    + file
                    + "__gmu_"
                    + str(self.params["gmu"])
                    + "__sigma_"
                    + str(self.params["sigma"])
                    + ".csv",
                    index=False,
                )

    def get_synthesis(self):
        """
        处理 data 路径下的 synthesis 数据集，将实验所需的数据集存放到 experiment 下的 synthesis 文件下(创建对应的数据集名命名的文件夹)
        在添加噪声时候，有两种情况，默认第一种，不添加噪声
        不添加噪声 => "noise_type": 0
        可以在 data 下的 synthesis 数据中添加高噪声 => "noise_type": 1
        """
        """判断是否有 noise_type 参数"""
        if "noise_type" not in self.params.keys():
            self.params["noise_type"] = 0

        """data 下的 synthesis 数据集均已经规整，归一化"""
        dir_path = DATA_PATH + "data/" + self.path + "/"
        for file in os.listdir(dir_path):
            raw_data = pandas.read_csv(dir_path + file)
            """创建以该文件名命名的文件夹"""
            file_name = os.path.splitext(file)[0]
            if not os.path.isdir(
                DATA_PATH + "experiment/" + self.path + "/" + file_name
            ):
                os.mkdir(DATA_PATH + "experiment/" + self.path + "/" + file_name)

            if self.params["noise_type"] == 0:
                """默认不添加噪声，读取 data 路径下的 synthesis 数据集并重新存入到 experiment 下的 synthesis 文件"""
                raw_data.to_csv(
                    DATA_PATH
                    + "experiment/"
                    + self.path
                    + "/"
                    + file_name
                    + "/"
                    + file_name
                    + ".csv",
                    index=False,
                )
            elif self.params["noise_type"] == 1:
                """添加高斯噪声"""
                col_list = list(raw_data.columns)
                """归一化"""
                noise_data = (
                    noramlized(self.params, self.add_noise(raw_data[col_list[0:-1]]))
                    .round(3)
                    .join(raw_data[col_list[-1]])
                )
                noise_data.to_csv(
                    DATA_PATH
                    + "experiment/"
                    + self.path
                    + "/"
                    + file_name
                    + "/"
                    + file_name
                    + "__gmu_"
                    + str(self.params["gmu"])
                    + "__sigma_"
                    + str(self.params["sigma"])
                    + ".csv",
                    index=False,
                )

    def get_uci(self):
        """
        处理 data 路径下的 uci 数据集，将实验所需的数据集存放到 experiment 下的 uci 文件下(创建对应的数据集名命名的文件夹)
        在添加噪声时候，有两种情况，默认第一种，不添加噪声
        不添加噪声 => "noise_type": 0
        可以在 data 下的 uci 数据中添加高噪声 => "noise_type": 1
        """
        """判断是否有 noise_type 参数"""
        if "noise_type" not in self.params.keys():
            self.params["noise_type"] = 0

        """data 下的 uci 数据集均已经规整，归一化"""
        dir_path = DATA_PATH + "data/" + self.path + "/"
        for file in os.listdir(dir_path):
            raw_data = pandas.read_csv(dir_path + file)
            """创建以该文件名命名的文件夹"""
            file_name = os.path.splitext(file)[0]
            if not os.path.isdir(
                DATA_PATH + "experiment/" + self.path + "/" + file_name
            ):
                os.mkdir(DATA_PATH + "experiment/" + self.path + "/" + file_name)

            if self.params["noise_type"] == 0:
                """默认不添加噪声，读取 data 路径下的 uci 数据集并重新存入到 experiment 下的 uci 文件"""
                raw_data.to_csv(
                    DATA_PATH
                    + "experiment/"
                    + self.path
                    + "/"
                    + file_name
                    + "/"
                    + file_name
                    + ".csv",
                    index=False,
                )
            elif self.params["noise_type"] == 1:
                """添加高斯噪声"""
                col_list = list(raw_data.columns)
                """归一化"""
                noise_data = (
                    noramlized(self.params, self.add_noise(raw_data[col_list[0:-1]]))
                    .round(3)
                    .join(raw_data[col_list[-1]])
                )
                noise_data.to_csv(
                    DATA_PATH
                    + "experiment/"
                    + self.path
                    + "/"
                    + file_name
                    + "/"
                    + file_name
                    + "__gmu_"
                    + str(self.params["gmu"])
                    + "__sigma_"
                    + str(self.params["sigma"])
                    + ".csv",
                    index=False,
                )

    def get_image(self):
        """
        处理 data 路径下的 image 数据集，将实验所需的数据集存放到 experiment 下的 image 文件下(创建对应的数据集名命名的文件夹)
        """
        pass
