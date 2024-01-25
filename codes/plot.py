# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
from sklearn.datasets import *
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from multiprocessing.pool import Pool
from .analyze import *

"""
绘图
"""
"""图片设置"""
font_ = {
    "family": "Times New Roman",
    "color": "black",
    "size": 16,
}


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

    def show_circles(self):
        """
        展示 demo 下的 circles 数据集结果
        """
        """为保存的图片创建个文件夹"""
        if not os.path.isdir(self.save_path + "circles/"):
            os.mkdir(self.save_path + "circles/")

        """绘图"""
        fig, axes = plt.subplots(6, 6, figsize=(18, 18))
        font_ = {
            "family": "Times New Roman",
            "color": "black",
            "size": 16,
        }

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
                    axes[5][j].set_xlabel(self.cicrles_title_list[j], fontdict=font_)
                if j == 0:
                    axes[i][0].set_ylabel(
                        "Noise level = "
                        + str(self.params["circles"]["noise"][i])
                        + "                                ",
                        rotation=0,
                        fontdict=font_,
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
        font_ = {
            "family": "Times New Roman",
            "color": "black",
        }

        for i in range(len(self.moons_title_list)):
            show_data = numpy.array(pandas.read_csv(path_list[i]))
            axes[i].scatter(show_data[:, 0], show_data[:, 1], c=label, s=7, marker=".")
            axes[i].axis("off")
            # axes[i].set_title(self.moons_title_list[i], y=-0.2)
            axes[i].set_title(self.moons_title_list[i], font_)

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


class PlotSynthesis:
    """
    处理 Synthesis 数据集相关的绘图
    1. 根据 ./result/demo/analyze/ 下的数据集绘制图，将图存放到 ./result/demo/plot/
    """

    def rho_compare(self):
        """
        改进局部密度的 DPC 算法与经典 DPC 算法在局部密度与聚类结果的对比图
        """

    def show_cluster_results(self):
        """
        展示 Synthesis 数据集上每个聚类算法的聚类结果
        """


class PlotUci:
    """
    处理 Uci 数据集相关的绘图
    1. 参数验证(也涉及到 Synthesis 部分的数据集)
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

    def show_improve_rho(self):
        """
        对比局部密度的改进
        """
        """字体配置文件"""
        font_ = {
            "family": "Times New Roman",
            "color": "black",
            "size": 16,
        }

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
        font_["color"] = "blue"
        plt.text(
            2,
            2,
            r"A",
            fontdict=font_,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.scatter(4, 4, s=30, c="g", marker="x")

        font_["color"] = "green"
        plt.text(
            4,
            4,
            r"B",
            fontdict=font_,
            verticalalignment="top",
            horizontalalignment="left",
        )

        """在密度中心画圈"""
        circle2 = plt.Circle((2, 2), 1, color="b", alpha=0.7, fill=False)
        circle4 = plt.Circle((4, 4), 1, color="g", alpha=0.7, fill=False)
        plt.gcf().gca().add_artist(circle2)
        plt.gcf().gca().add_artist(circle4)

        """突出 A 的邻居"""
        font_["color"] = "blue"
        font_["size"] = 10
        plt.text(
            2,
            3,
            r"$A_1$",
            fontdict=font_,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            1,
            2,
            r"$A_2$",
            fontdict=font_,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            2,
            1,
            r"$A_3$",
            fontdict=font_,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            3,
            2,
            r"$A_4$",
            fontdict=font_,
            verticalalignment="top",
            horizontalalignment="left",
        )

        """突出 B 的邻居"""
        font_["color"] = "green"
        font_["size"] = 10
        plt.text(
            4,
            5,
            r"$B_1$",
            fontdict=font_,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            3,
            4,
            r"$B_2$",
            fontdict=font_,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            4,
            3,
            r"$B_3$",
            fontdict=font_,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            5,
            4,
            r"$B_4$",
            fontdict=font_,
            verticalalignment="top",
            horizontalalignment="left",
        )

        """加点标注"""
        font_["color"] = "black"
        font_["size"] = 14
        plt.text(
            5,
            1,
            r"$k=4$",
            fontdict=font_,
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
        font_["color"] = "green"
        plt.text(
            TWO_STEP_CLUSTERS["cluster1"][0, 0],
            TWO_STEP_CLUSTERS["cluster1"][0, 1],
            r"center",
            fontdict=font_,
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
        font_["color"] = "red"
        plt.text(
            TWO_STEP_CLUSTERS["cluster2"][0, 0],
            TWO_STEP_CLUSTERS["cluster2"][0, 1],
            r"center",
            fontdict=font_,
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
        font_["color"] = "blue"
        plt.text(
            TWO_STEP_CLUSTERS["cluster3"][0, 0],
            TWO_STEP_CLUSTERS["cluster3"][0, 1],
            r"center",
            fontdict=font_,
            verticalalignment="top",
            horizontalalignment="left",
        )

    def two_step_assign1(self):
        """
        两步样本分配策略演示图，第一张图
        """
        self.assign_base()
        """标注"""
        font_["color"] = "black"
        font_["size"] = 16
        plt.text(
            10,
            11,
            r"$\bigtriangleup$ = center",
            fontdict=font_,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10,
            10,
            r"$\cdot$ = point",
            fontdict=font_,
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
        font_["color"] = "red"
        font_["size"] = 12
        plt.text(
            5,
            6.4,
            r"$c_1$",
            fontdict=font_,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.scatter(8, 6.5, s=24, c="blue", marker="p")
        font_["color"] = "blue"
        font_["size"] = 12
        plt.text(
            8,
            6.5,
            r"$c_2$",
            fontdict=font_,
            verticalalignment="top",
            horizontalalignment="left",
        )

        circle2 = plt.Circle((5, 6.4), 1.3, color="r", alpha=0.7, fill=False)
        circle4 = plt.Circle((8, 6.5), 1.3, color="b", alpha=0.7, fill=False)
        plt.gcf().gca().add_artist(circle2)
        plt.gcf().gca().add_artist(circle4)

        """标注"""
        font_["color"] = "black"
        font_["size"] = 16
        plt.text(
            10,
            11,
            r"k = 5",
            fontdict=font_,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10,
            10,
            r"$\bigtriangleup$ = center",
            fontdict=font_,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10,
            9,
            r"$\cdot$ = point",
            fontdict=font_,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10,
            8,
            r"$\star$ = density peak",
            fontdict=font_,
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
        font_["color"] = "blue"
        font_["size"] = 12
        plt.text(
            8,
            6.5,
            r"$c_2$",
            fontdict=font_,
            verticalalignment="top",
            horizontalalignment="left",
        )

        """[8, 6.5], [7, 6], [5.8, 5.8]"""
        plt.scatter(7, 6, s=10, c="blue", marker="x")
        plt.scatter(5.8, 5.8, s=10, c="blue", marker="x")

        """标注"""
        font_["color"] = "black"
        font_["size"] = 16
        plt.text(
            10,
            11,
            r"k = 5",
            fontdict=font_,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10,
            10,
            r"$\bigtriangleup$ = center",
            fontdict=font_,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10,
            9,
            r"$+$ = hard label",
            fontdict=font_,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10,
            8,
            r"$\times$ = soft label",
            fontdict=font_,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10,
            7,
            r"$\star$ = density peak",
            fontdict=font_,
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
        font_["color"] = "black"
        font_["size"] = 16
        plt.text(
            10,
            11,
            r"$\bigtriangleup$ = center",
            fontdict=font_,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10,
            10,
            r"$+$ = label",
            fontdict=font_,
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
