# -*- coding: utf-8 -*-


import os
import shutil
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
        self.save_path = self.path.replace("/demo/result", "/demo/analyze")

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


def analyze_data_algorithm(dir_path, data_name, algorithm_name):
    """
    遍历 dir_path 下的 data_name 数据集的 algorithm_name 算法文件夹下的所有文件，获取最优的结果
    将该文件复制到 dir_path/analyze + data_name 下的 algorithm 文件
    同时，收集所有的 ACC，AMI，ARI，FMI，NMI 结果
    Args:
        dir_path (_type_): 当前文件夹
        data_name (_type_): 数据集名称
        algorithm_name (_type_): 算法名称
    """
    """判断 dir_path/analyze 下有没有 dataset_name 文件下"""
    if not os.path.isdir(dir_path + "/analyze/" + data_name):
        os.mkdir(dir_path + "/analyze/" + data_name)

    """文件结果列表"""
    result_list = os.listdir(dir_path + "result/" + data_name + "/" + algorithm_name)
    """记录最优结果"""
    best_result_file = ""
    """收集所有结果"""
    all_result = {
        "ACC": list(),
        "AMI": list(),
        "ARI": list(),
        "NMI": list(),
        "FMI": list(),
        "FILE_NAME": list(),
        "cluster_acc": -1,
        "adjusted_mutual_info": -1,
        "adjusted_rand_index": -1,
        "fowlkes_mallows_index": -1,
        "normalized_mutual_info": -1,
        "file_name": "",
    }

    """遍历所有文件"""
    for file in result_list:
        """读取json"""
        with open(
            dir_path + "result/" + data_name + "/" + algorithm_name + "/" + file, "r"
        ) as f:
            data = json.load(f)

        """file name"""
        all_result["FILE_NAME"].append(file)

        """ACC"""
        all_result["ACC"].append(data["cluster_acc"])
        if data["cluster_acc"] > all_result["cluster_acc"]:
            all_result["cluster_acc"] = data["cluster_acc"]
            best_result_file = file

        """AMI"""
        all_result["AMI"].append(data["adjusted_mutual_info"])
        if data["adjusted_mutual_info"] > all_result["adjusted_mutual_info"]:
            all_result["adjusted_mutual_info"] = data["adjusted_mutual_info"]
            best_result_file = file

        """ARI"""
        all_result["ARI"].append(data["adjusted_rand_index"])
        if data["adjusted_rand_index"] > all_result["adjusted_rand_index"]:
            all_result["adjusted_rand_index"] = data["adjusted_rand_index"]
            best_result_file = file

        """FMI"""
        all_result["FMI"].append(data["fowlkes_mallows_index"])
        if data["fowlkes_mallows_index"] > all_result["fowlkes_mallows_index"]:
            all_result["fowlkes_mallows_index"] = data["fowlkes_mallows_index"]
            best_result_file = file

        """NMI"""
        all_result["NMI"].append(data["normalized_mutual_info"])
        if data["normalized_mutual_info"] > all_result["normalized_mutual_info"]:
            all_result["normalized_mutual_info"] = data["normalized_mutual_info"]
            best_result_file = file

    """保存结果"""
    all_result["file_name"] = best_result_file
    with open(
        dir_path + "analyze/" + data_name + "/" + algorithm_name + "_all.json",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(json.dumps(all_result, ensure_ascii=False))
    """移动文件"""
    shutil.copyfile(
        dir_path
        + "result/"
        + data_name
        + "/"
        + algorithm_name
        + "/"
        + best_result_file,
        dir_path
        + "analyze/"
        + data_name
        + "/"
        + algorithm_name
        + "+"
        + best_result_file,
    )


class AnalyzeSynthesis:
    """
    分析 synthesis 数据集，根据数据集的实验结果选择最佳的结果，将文件保存到 ./result/
    """

    def __init__(self, path="./result/synthesis/result/") -> None:
        """
        初始化相关成员
        Args:
            path (str, optional): 文件路径，由于调用该类的主程序路径待定，这里使用手动传入，同时根据该路径解析出保存路径. Defaults to "./result/synthesis/result/".
        """
        """文件路径"""
        self.path = path
        """保存结果路径"""
        self.save_path = self.path.replace("/synthesis/result", "/synthesis")

    def analyze_synthesis(self):
        """
        遍历 synthesis 下所有的数据集的所有算法文件夹下的所有文件，获取最优的结果
        将该文件复制到 self.path/analyze/数据集名下的算法文件中
        """
        data_path = self.path.split("result/")[0] + "dataset/experiment/synthesis/"

        """遍历数据集"""
        for data_name in os.listdir(self.path):
            """遍历算法"""
            for algorithm_name in os.listdir(self.path + data_name):
                """获取最优结果"""
                analyze_data_algorithm(self.save_path, data_name, algorithm_name)

            """移动原始数据集"""
            shutil.copy(
                data_path + data_name + "/" + data_name + ".csv",
                self.save_path + "analyze/" + data_name + ".csv",
            )


class AnalyzeUci:
    """
    分析 UCI 数据集，根据数据集的实验结果选择最佳的结果，将文件保存到 ./result/uci/analyze/
    """

    def __init__(self, path="./result/uci/result/") -> None:
        """
        初始化相关成员
        Args:
            path (str, optional): 文件路径，由于调用该类的主程序路径待定，这里使用手动传入，同时根据该路径解析出保存路径. Defaults to "./result/uci/result/".
        """
        """文件路径"""
        self.path = path
        """保存结果路径"""
        self.save_path = self.path.replace("/uci/result", "/uci")

    def analyze_uci(self):
        """
        遍历 uci 下所有的数据集的所有算法文件夹下的所有文件，获取最优的结果
        将该文件复制到 self.path/analyze/数据集名下的算法文件中
        """
        data_path = self.path.split("result/")[0] + "dataset/experiment/uci/"

        """遍历数据集"""
        for data_name in os.listdir(self.path):
            """遍历算法"""
            for algorithm_name in os.listdir(self.path + data_name):
                """获取最优结果"""
                analyze_data_algorithm(self.save_path, data_name, algorithm_name)

            """移动原始数据集"""
            shutil.copy(
                data_path + data_name + "/" + data_name + ".csv",
                self.save_path + "analyze/" + data_name + ".csv",
            )

    def dimensionality_reduction(self):
        """
        对 UCI 数据集进行降维，便于后续作图
        """


class AnalyzeImage:
    """
    分析 image 数据集，根据数据集的实验结果选择最佳的结果，将文件保存到 ./result/image/analyze/
    """

    def __init__(self, path="./result/image/result/") -> None:
        """
        初始化相关成员
        Args:
            path (str, optional): 文件路径，由于调用该类的主程序路径待定，这里使用手动传入，同时根据该路径解析出保存路径. Defaults to "./result/image/result/".
        """
        """文件路径"""
        self.path = path
        """保存结果路径"""
        self.save_path = self.path.replace("/image/result", "/image")

    def analyze_image(self):
        """
        遍历 image 下所有的数据集的所有算法文件夹下的所有文件，获取最优的结果
        将该文件复制到 self.path/analyze/数据集名下的算法文件中
        """
        data_path = self.path.split("result/")[0] + "dataset/experiment/image/"

        """遍历数据集"""
        for data_name in os.listdir(self.path):
            """test"""
            if data_name in ["coil20", "jaffe"]:
                """遍历算法"""
                for algorithm_name in os.listdir(self.path + data_name):
                    """获取最优结果"""
                    analyze_data_algorithm(self.save_path, data_name, algorithm_name)

                """移动原始数据集"""
                shutil.copy(
                    data_path + data_name + "/" + data_name + ".csv",
                    self.save_path + "analyze/" + data_name + ".csv",
                )
                """移动 mat 文件"""
                shutil.copy(
                    "./dataset/experiment/image/"
                    + data_name
                    + "/"
                    + data_name
                    + ".mat",
                    self.save_path + "analyze/" + data_name + ".mat",
                )
