# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
from sklearn.datasets import *
from .analyze import *

"""
绘图
"""


class PlotDemo:
    """
    处理 Demo 数据集相关的绘图
    1. 根据 ./result/demo/analyze/ 下的数据集绘制图，将图存放到 ./result/demo/plot/
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
        """字体配置"""
        self.font = {
            "family": "Times New Roman",
            "color": "black",
            "size": 16,
        }

    def show_circles(self):
        """
        展示 demo 下的 circles 数据集结果
        """
        """为保存的图片创建个文件夹"""
        if not os.path.isdir(self.save_path + "circles/"):
            os.mkdir(self.save_path + "circles/")

        """绘图"""
        fig, axes = plt.subplots(6, 6, figsize=(18, 18))

        for i in range(len(self.params["circles"]["noise"])):
            """不同噪声级别"""
            dir_path = (
                self.path
                + "circles/noise_"
                + str(self.params["circles"]["noise"][i])
                + "/num_"
            )
            label = numpy.array(
                pandas.read_csv(dir_path + str(self.params["circles"]["num"]) + ".csv")
            )[:, -1]

            for j in range(len(self.cicrles_title_list)):
                """不同度量"""
                show_data = numpy.array(
                    pandas.read_csv(
                        dir_path
                        + str(self.params["circles"]["num"])
                        + "__"
                        + DIS_METHOD[j]
                        + ".csv"
                    )
                )
                axes[i][j].scatter(
                    show_data[:, 0], show_data[:, 1], c=label, s=7, marker="."
                )
                axes[i][j].set_xticks([])
                axes[i][j].set_yticks([])

                if i == 5:
                    axes[5][j].set_xlabel(
                        self.cicrles_title_list[j], fontdict=self.font
                    )
                if j == 0:
                    axes[i][0].set_ylabel(
                        "Noise level = "
                        + str(self.params["circles"]["noise"][i])
                        + "                                ",
                        rotation=0,
                        fontdict=self.font,
                    )

        plt.tight_layout()
        plt.savefig(
            self.save_path
            + "circles/num_"
            + str(self.params["circles"]["num"])
            + "__noise_"
            + str(self.params["circles"]["noise"])
            + ".svg",
            bbox_inches="tight",
        )
        plt.show()

    def show_moons(self):
        """
        展示 demo 下的 moons 数据集结果
        """
        """根据参数选择本次作图需要的数据集"""
        path_list = [
            self.path + "moons/num_" + str(self.params["moons"]["num"]) + ".csv",
            self.path + "moons/num_" + str(self.params["moons"]["num"]) + "__noise.csv",
        ]
        """label"""
        label = numpy.array(pandas.read_csv(path_list[0]))[:, -1]

        for dis_name in DIS_METHOD:
            file_path = (
                self.path
                + "moons/noise_"
                + str(self.params["moons"]["noise"])
                + "/num_"
                + str(self.params["moons"]["num"])
                + "__"
                + dis_name
                + ".csv"
            )
            if os.path.isfile(file_path):
                path_list.append(file_path)

        """为保存的图片创建个文件夹"""
        if not os.path.isdir(self.save_path + "moons/"):
            os.mkdir(self.save_path + "moons/")

        """绘图"""
        fig, axes = plt.subplots(1, 8, figsize=(16, 2))

        for i in range(len(self.moons_title_list)):
            show_data = numpy.array(pandas.read_csv(path_list[i]))
            axes[i].scatter(show_data[:, 0], show_data[:, 1], c=label, s=7, marker=".")
            axes[i].axis("off")
            axes[i].set_title(self.moons_title_list[i], self.font)

        plt.tight_layout()
        plt.savefig(
            self.save_path
            + "moons/num_"
            + str(self.params["moons"]["num"])
            + "__noise_"
            + str(self.params["moons"]["noise"])
            + ".svg",
            bbox_inches="tight",
        )
        plt.show()


def show_data_algorithm(path, data_name, algorithm_name):
    """
    绘制一个数据集上一个算法的聚类结果图
    Args:
        path (_type_): 工作目录
        data_name (_type_): 数据集名称
        algorithm_list (_type_): 算法名称
    """
    """原始数据"""
    data = pandas.read_csv(path + "analyze/" + data_name + ".csv")
    """寻找 algorithm_name 对应的文件"""
    file_name = ""
    for file in os.listdir(path + "analyze/" + data_name):
        if file.find(algorithm_name + "+") == 0:
            file_name = file

    """预测结果"""
    with open(path + "analyze/" + data_name + "/" + file_name, "r") as f:
        pred_result = json.load(f)
    """将预测的标签写入到原始数据中"""
    data["label"] = pred_result["label"]


class PlotSynthesis:
    """
    处理 Synthesis 数据集相关的绘图
    1. 根据 ./result/synthesis/analyze/ 下的数据集绘制图，将图存放到 ./result/demo/plot/
    """

    def __init__(self, path="./result/synthesis/", dpc_algorithm="dpc_irho") -> None:
        """
        初始化相关成员
        Args:
            save_path (str, optional): 工作目录. Defaults to "./result/synthesis/".
        """
        """保存结果路径"""
        self.path = path
        """字体配置"""
        self.font = {
            "family": "Times New Roman",
            "color": "black",
            "size": 16,
        }
        """数据列表"""
        self.dataset_list = [
            # "aggregation",
            # "compound",
            # "D31",
            # "flame",
            "jain",
            # "pathbased",
            # "R15",
            # "S2",
            "spiral",
        ]
        """算法列表"""
        self.algorithm_list = [
            "agglomerativeClustering",
            "kmeans",
            "dbscan",
            "optics",
            "spectralClustering",
            "dpc",
            "dpc_knn",
            "snn_dpc",
        ]
        """添加改进算法"""
        self.algorithm_list.append(dpc_algorithm)

    def gain_cluster_rho(self, data_name, algorithm_name):
        """
        拆分 self.rho_cluster
        获取局部密度
        Args:
            data_name (_type_): _description_
            algorithm_name (_type_): _description_
        Returns:
            rho_value (_type_): 局部密度
            file_name (_type_): 文件名
        """
        """遍历文件夹，获取文件名"""
        file_name = ""
        """局部密度结果"""
        rho_value = None
        for file in os.listdir(self.path + "analyze/" + data_name):
            if "+" in file:
                file_split = file.split("+")
                if file_split[0] == algorithm_name:
                    file_name = file_split[1]

        """数据集相关的基本信息"""
        path = SYNTHESIS_PARAMS[data_name]["path"]
        save_path = SYNTHESIS_PARAMS[data_name]["save_path"]
        num = SYNTHESIS_PARAMS[data_name]["num"]

        """解析文件名，获取参数，重新运行对应算法，获取局部密度"""
        if algorithm_name == "dpc":
            """解析出 percent"""
            _percent = 0
            for item in file_name.split(".json")[0].split("__"):
                if "dcp" in item:
                    _percent = item.split("_")[1]

            al = Dpc(
                path,
                save_path,
                num,
                _percent,
            )
            rho_value = al.rho
        elif algorithm_name == "dpc_irho":
            """解析出 k 与 mu"""
            _param = {"k": 0, "mu": 0}
            for item in file_name.split(".json")[0].split("__"):
                if "k" in item:
                    _param["k"] = int(item.split("_")[1])
                if "mu" in item:
                    _param["mu"] = int(item.split("_")[1])

            al = DpcIRho(
                path,
                save_path,
                num,
                1,
                1,
                0,
                0,
                "ckrod",
                [],
                False,
                _param,
            )
            rho_value = al.rho

        return rho_value, file_name

    def rho_cluster(self, axes, data_name, algorithm_name):
        """
        绘制一个算法在一个数据集上的局部密度图与聚类结果图
        Args:
            axes (_type_): 绘图句柄
            data_name (_type_): 文件名
            algorithm_name (_type_): 算法名称
        """
        """原始数据"""
        data = pandas.read_csv(self.path + "analyze/" + data_name + ".csv")
        col = list(data.columns)
        """分离原始数据与原始标签"""
        origin_data = data.loc[:, col[0:-1]]

        """获取局部密度"""
        rho, file_name = self.gain_cluster_rho(data_name, algorithm_name)

        """算法预测结果"""
        with open(
            self.path + "analyze/" + data_name + "/" + algorithm_name + "+" + file_name,
            "r",
        ) as f:
            al_data = json.load(f)
        """算法预测的标签"""
        al_label = al_data["label"]
        """算法预测的中心"""
        al_center = al_data["center"]

        """绘图"""
        colors = dict()
        numpy.random.seed(1)

        """根据聚类结果拆分数据，得到一个字典，键为类别，值为归属该类的样本索引"""
        al_cluster_result = dict()
        for k in set(al_label):
            colors[k] = numpy.random.rand(3).reshape(1, -1)
            al_cluster_result[k] = list()

        for index in range(len(al_label)):
            al_cluster_result[al_label[index]].append(index)

        index = 0
        for k, v in al_cluster_result.items():
            """局部密度图"""
            axes[0].scatter(v, rho[v], c=colors[k], s=15, marker=".")
            axes[0].scatter(
                al_center[index], rho[al_center[index]], c=colors[k], s=128, marker="*"
            )

            """图像设置"""
            axes[0].spines["right"].set_color("none")
            axes[0].spines["top"].set_color("none")
            axes[0].set_xlim(0, len(al_label) * 1.1)
            axes[0].set_ylim(0, max(rho) * 1.1)
            axes[0].grid(linestyle="-", linewidth=0.5)
            axes[0].set_xlabel("num")
            axes[0].set_ylabel(r"$\rho$")

            self.font["size"] = 16

            if algorithm_name == "dpc":
                axes[0].set_title(r"$\rho$ value of traditional DPC", self.font)
                axes[1].set_title(r"Clustering result of traditional DPC", self.font)
            else:
                axes[0].set_title(r"$\rho$ value of LW-DPC", self.font)
                axes[1].set_title(r"Clustering result of LW-DPC", self.font)

            """聚类结果"""
            axes[1].scatter(
                origin_data.loc[v, col[0]],
                origin_data.loc[v, col[1]],
                c=colors[k],
                s=15,
                marker=".",
            )
            axes[1].scatter(
                origin_data.loc[al_center[index], col[0]],
                origin_data.loc[al_center[index], col[1]],
                c=colors[k],
                s=128,
                marker="*",
            )

            """图像设置"""
            axes[1].spines["right"].set_color("none")
            axes[1].spines["top"].set_color("none")
            axes[1].set_xlim(0, 1.05)
            axes[1].set_ylim(0, 1.05)
            axes[1].set_xticks(numpy.linspace(0, 1, 11))
            axes[1].set_yticks(numpy.linspace(0, 1, 11))
            axes[1].set_aspect(1)
            axes[1].grid(linestyle="-", linewidth=0.5)
            axes[1].set_xlabel("X")
            axes[1].set_ylabel("Y")

            index += 1

    def show_rho_compare(self):
        """
        使用的数据集为 jain
        改进局部密度的 DPC 算法与经典 DPC 算法在局部密度与聚类结果的对比图
        """
        data_name = "jain"
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        self.rho_cluster(axes[0], data_name, "dpc")
        self.rho_cluster(axes[1], data_name, "dpc_irho")

        """保存图片"""
        plt.tight_layout()
        plt.savefig(self.path + "plot/compare_rho.svg", bbox_inches="tight")
        plt.show()

    def show_cluster_results(self):
        """
        展示 Synthesis 数据集上每个聚类算法的聚类结果
        """
        """不同数据集"""
        for data_name in self.dataset_list:
            """3 x 3 的格式"""
            fig, axes = plt.subplots(3, 3, figsize=(18, 18))
            for i in range(3):
                for j in range(3):
                    """"""
                    show_data_algorithm(
                        self.path, data_name, self.algorithm_list[i * 3 + j]
                    )

            """保存图片"""
            # plt.tight_layout()
            # plt.savefig(self.path + "plot/" + data_name + ".svg", bbox_inches="tight")
            # plt.show()


class PlotUci:
    """
    处理 Uci 数据集相关的绘图
    1. 根据 ./result/uci/analyze/ 下的数据集绘制图，将图存放到 ./result/uci/plot/，需要降维
    2. 参数验证(也涉及到 Synthesis 部分的数据集)
    """

    def param_argue_k(self):
        """
        对 k 进行讨论
        """

    def param_argue_mu(self):
        """
        对 mu 进行讨论
        """

    def param_order_sensitivity(self):
        """
        对样本顺序进行讨论
        """


class PlotImage:
    """
    处理 Image 数据集相关的绘图
    1. 主要是聚类结果展示
    """

    def __init__(self, save_path="./result/diagram/") -> None:
        """
        初始化相关成员
        Args:
            save_path (str, optional): 保存结果的路径. Defaults to "./result/diagram/".
        """
        """保存结果路径"""
        self.save_path = save_path
        """字体配置"""
        self.font = {
            "family": "Times New Roman",
            "color": "black",
            "size": 16,
        }


class MyPlot:
    """
    这个类主要绘制与数据集无关的图
    """

    def __init__(self, save_path="./result/diagram/") -> None:
        """
        初始化相关成员
        Args:
            save_path (str, optional): 保存结果的路径. Defaults to "./result/diagram/".
        """
        """保存结果路径"""
        self.save_path = save_path
        """字体配置"""
        self.font = {
            "family": "Times New Roman",
            "color": "black",
            "size": 16,
        }

    def show_improve_rho(self):
        """
        对比局部密度的改进
        """
        plt.scatter(RHO_COMPARE[:, 0], RHO_COMPARE[:, 1], s=24, c="r", marker=".")
        ax = plt.gca()
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")

        plt.xlim(-0.5, 6.5)
        plt.ylim(-0.5, 6.5)
        ax.set_aspect(1)
        plt.grid(linestyle="-", linewidth=0.5)

        """演示"""
        """两个密度中心"""
        plt.scatter(2, 2, s=30, c="b", marker="*")
        self.font["color"] = "blue"
        plt.text(
            2,
            2,
            r"A",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.scatter(4, 4, s=30, c="g", marker="x")

        self.font["color"] = "green"
        plt.text(
            4,
            4,
            r"B",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )

        """在密度中心画圈"""
        circle2 = plt.Circle((2, 2), 1, color="b", alpha=0.7, fill=False)
        circle4 = plt.Circle((4, 4), 1, color="g", alpha=0.7, fill=False)
        plt.gcf().gca().add_artist(circle2)
        plt.gcf().gca().add_artist(circle4)

        """突出 A 的邻居"""
        self.font["color"] = "blue"
        self.font["size"] = 10
        plt.text(
            2,
            3,
            r"$A_1$",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            1,
            2,
            r"$A_2$",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            2,
            1,
            r"$A_3$",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            3,
            2,
            r"$A_4$",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )

        """突出 B 的邻居"""
        self.font["color"] = "green"
        self.font["size"] = 10
        plt.text(
            4,
            5,
            r"$B_1$",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            3,
            4,
            r"$B_2$",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            4,
            3,
            r"$B_3$",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            5,
            4,
            r"$B_4$",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )

        """加点标注"""
        self.font["color"] = "black"
        self.font["size"] = 14
        plt.text(
            5,
            1,
            r"$k=4$",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )

        plt.tight_layout()
        plt.savefig(self.save_path + "rho_compare.svg", bbox_inches="tight")
        plt.show()

    def domino_effect(self):
        """
        一步分配策略中的多米诺效应演示图，可以用现成数据集，但不用跟别人的重复
        """

    def assign_base(self):
        """
        两步分配策略蓝本演示图
        Args:
        """
        plt.axis("off")
        ax = plt.gca()
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")

        plt.xlim(0, 13)
        plt.ylim(0, 13)
        ax.set_aspect(1)

        """cluster1"""
        plt.scatter(
            TWO_STEP_CLUSTERS["cluster1"][:, 0],
            TWO_STEP_CLUSTERS["cluster1"][:, 1],
            s=18,
            c="green",
            marker=".",
        )
        plt.scatter(
            TWO_STEP_CLUSTERS["cluster1"][0, 0],
            TWO_STEP_CLUSTERS["cluster1"][0, 1],
            s=36,
            c="green",
            marker="^",
        )
        self.font["color"] = "green"
        plt.text(
            TWO_STEP_CLUSTERS["cluster1"][0, 0],
            TWO_STEP_CLUSTERS["cluster1"][0, 1],
            r"center",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )

        """cluster2"""
        plt.scatter(
            TWO_STEP_CLUSTERS["cluster2"][:, 0],
            TWO_STEP_CLUSTERS["cluster2"][:, 1],
            s=18,
            c="red",
            marker=".",
        )
        plt.scatter(
            TWO_STEP_CLUSTERS["cluster2"][0, 0],
            TWO_STEP_CLUSTERS["cluster2"][0, 1],
            s=36,
            c="red",
            marker="^",
        )
        self.font["color"] = "red"
        plt.text(
            TWO_STEP_CLUSTERS["cluster2"][0, 0],
            TWO_STEP_CLUSTERS["cluster2"][0, 1],
            r"center",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )

        """cluster3"""
        plt.scatter(
            TWO_STEP_CLUSTERS["cluster3"][:, 0],
            TWO_STEP_CLUSTERS["cluster3"][:, 1],
            s=18,
            c="blue",
            marker=".",
        )
        plt.scatter(
            TWO_STEP_CLUSTERS["cluster3"][0, 0],
            TWO_STEP_CLUSTERS["cluster3"][0, 1],
            s=36,
            c="blue",
            marker="^",
        )
        self.font["color"] = "blue"
        plt.text(
            TWO_STEP_CLUSTERS["cluster3"][0, 0],
            TWO_STEP_CLUSTERS["cluster3"][0, 1],
            r"center",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )

    def two_step_assign1(self):
        """
        两步样本分配策略演示图，第一张图
        """
        self.assign_base()
        """标注"""
        self.font["color"] = "black"
        self.font["size"] = 16
        plt.text(
            10,
            11,
            r"$\bigtriangleup$ = center",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10,
            10,
            r"$\cdot$ = point",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )

        """保存图片"""
        plt.tight_layout()
        plt.savefig(self.save_path + "two_step_assign1.svg", bbox_inches="tight")
        plt.show()

    def two_step_assign2(self):
        """
        两步样本分配策略演示图，第二张图
        """
        self.assign_base()
        """突出密度中心，在密度中心画圈"""
        plt.scatter(5, 6.4, s=24, c="red", marker="p")
        self.font["color"] = "red"
        self.font["size"] = 12
        plt.text(
            5,
            6.4,
            r"$c_1$",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.scatter(8, 6.5, s=24, c="blue", marker="p")
        self.font["color"] = "blue"
        self.font["size"] = 12
        plt.text(
            8,
            6.5,
            r"$c_2$",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )

        circle2 = plt.Circle((5, 6.4), 1.3, color="r", alpha=0.7, fill=False)
        circle4 = plt.Circle((8, 6.5), 1.3, color="b", alpha=0.7, fill=False)
        plt.gcf().gca().add_artist(circle2)
        plt.gcf().gca().add_artist(circle4)

        """标注"""
        self.font["color"] = "black"
        self.font["size"] = 16
        plt.text(
            10,
            11,
            r"k = 5",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10,
            10,
            r"$\bigtriangleup$ = center",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10,
            9,
            r"$\cdot$ = point",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10,
            8,
            r"$\star$ = density peak",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )

        """保存图片"""
        plt.tight_layout()
        plt.savefig(self.save_path + "two_step_assign2.svg", bbox_inches="tight")
        plt.show()

    def two_step_assign3(self):
        """
        两步样本分配策略演示图，第三张图
        """
        self.assign_base()
        """cluster1"""
        plt.scatter(
            TWO_STEP_CLUSTERS["cluster1"][:, 0],
            TWO_STEP_CLUSTERS["cluster1"][:, 1],
            s=18,
            c="green",
            marker="+",
        )
        """cluster2"""
        plt.scatter(
            TWO_STEP_CLUSTERS["cluster2"][:, 0],
            TWO_STEP_CLUSTERS["cluster2"][:, 1],
            s=18,
            c="red",
            marker="+",
        )
        """cluster3"""
        plt.scatter(
            TWO_STEP_CLUSTERS["cluster3"][:, 0],
            TWO_STEP_CLUSTERS["cluster3"][:, 1],
            s=18,
            c="blue",
            marker="+",
        )

        """突出密度中心，在密度中心画圈"""
        plt.scatter(8, 6.5, s=24, c="blue", marker="p")
        circle4 = plt.Circle((8, 6.5), 1.3, color="b", alpha=0.7, fill=False)
        plt.gcf().gca().add_artist(circle4)
        self.font["color"] = "blue"
        self.font["size"] = 12
        plt.text(
            8,
            6.5,
            r"$c_2$",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )

        """[8, 6.5], [7, 6], [5.8, 5.8]"""
        plt.scatter(7, 6, s=10, c="blue", marker="x")
        plt.scatter(5.8, 5.8, s=10, c="blue", marker="x")

        """标注"""
        self.font["color"] = "black"
        self.font["size"] = 16
        plt.text(
            10,
            11,
            r"k = 5",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10,
            10,
            r"$\bigtriangleup$ = center",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10,
            9,
            r"$+$ = hard label",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10,
            8,
            r"$\times$ = soft label",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10,
            7,
            r"$\star$ = density peak",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )

        """保存图片"""
        plt.tight_layout()
        plt.savefig(self.save_path + "two_step_assign3.svg", bbox_inches="tight")
        plt.show()

    def two_step_assign4(self):
        """
        两步样本分配策略演示图，第四张图
        """
        self.assign_base()
        """cluster1"""
        plt.scatter(
            TWO_STEP_CLUSTERS["cluster1"][:, 0],
            TWO_STEP_CLUSTERS["cluster1"][:, 1],
            s=18,
            c="green",
            marker="+",
        )
        """cluster2"""
        plt.scatter(
            TWO_STEP_CLUSTERS["cluster2"][:, 0],
            TWO_STEP_CLUSTERS["cluster2"][:, 1],
            s=18,
            c="red",
            marker="+",
        )
        """cluster3"""
        plt.scatter(
            TWO_STEP_CLUSTERS["cluster3"][:, 0],
            TWO_STEP_CLUSTERS["cluster3"][:, 1],
            s=18,
            c="blue",
            marker="+",
        )

        """纠错 [8, 6.5], [7, 6], [5.8, 5.8]"""
        plt.scatter(7, 6, s=18, c="red", marker="+")
        plt.scatter(5.8, 5.8, s=18, c="red", marker="+")
        plt.scatter(8, 6.5, s=18, c="red", marker="+")

        """标注"""
        self.font["color"] = "black"
        self.font["size"] = 16
        plt.text(
            10,
            11,
            r"$\bigtriangleup$ = center",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10,
            10,
            r"$+$ = label",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )

        """保存图片"""
        plt.tight_layout()
        plt.savefig(self.save_path + "two_step_assign4.svg", bbox_inches="tight")
        plt.show()

    def two_step_assign(self):
        """
        两步样本分配策略演示图
        """
        """两步样本分配策略演示图，第一张图"""
        self.two_step_assign1()
        """两步样本分配策略演示图，第二张图"""
        self.two_step_assign2()
        """两步样本分配策略演示图，第三张图"""
        self.two_step_assign3()
        """两步样本分配策略演示图，第四张图"""
        self.two_step_assign4()
