# -*- coding: utf-8 -*-


import random
import matplotlib.pyplot as plt
from sklearn.datasets import *
from .analyze import *
from matplotlib import rcParams
from scipy.io import loadmat


"""
绘图类
"""
"""绘图标题名称"""
PLOT_TITLE = {
    "agglomerativeClustering": "AgglomerativeClustering",
    "kmeans": "K-Means",
    "dbscan": "Dbscan",
    "optics": "Optics",
    "spectralClustering": "SpectralClustering",
    "dpc": "DPC",
    "dpc_knn": "DPC-KNN",
    "snn_dpc": "SNN-DPC",
    "dpc_ckrod": "Dpc-Ckrod",
    "dpc_irho": "LW-DPC",
    "dpc_iass": "DPC-TSA",
}
PLOT_TITLE_NUM = [
    "(a) ",
    "(b) ",
    "(c) ",
    "(d) ",
    "(e) ",
    "(f) ",
    "(g) ",
    "(h) ",
    "(i) ",
]


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
            "Gaussian kernel",
            "Manhattan",
            "ROD",
            "KROD",
            "CKROD",
        ]
        self.moons_title_list = [
            "Original data",
            "Noise data",
            "Cosine",
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

    def show_moons(self):
        """
        展示 demo 下的 moons 数据集结果，3 x 3，对应图 3-1
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
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        numpy.random.seed(10)

        for i in range(3):
            for j in range(3):
                """绘图"""
                show_data = numpy.array(pandas.read_csv(path_list[i * 3 + j]))
                axes[i][j].scatter(
                    show_data[:, 0],
                    show_data[:, 1],
                    c=label,
                    cmap="winter",
                    # cmap="RdYlGn",
                    s=7,
                    marker=".",
                )
                axes[i][j].axis("off")
                axes[i][j].set_title(
                    PLOT_TITLE_NUM[i * 3 + j] + self.moons_title_list[i * 3 + j],
                    self.font,
                    y=-0.1,
                )

        plt.tight_layout()
        plt.savefig(
            self.save_path
            + "moons/num_"
            + str(self.params["moons"]["num"])
            + "__noise_"
            + str(self.params["moons"]["noise"])
            + ".pdf",
            bbox_inches="tight",
        )
        plt.show()

    def show_circles(self):
        """
        展示 demo 下的 circles 数据集结果，6 x 6，对应图 3-3
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
                        + DIS_METHOD[j + 1]
                        + ".csv"
                    )
                )
                gau_label = []
                if DIS_METHOD[j + 1] == "gau":
                    """label 换个顺序"""
                    for ele in label:
                        if ele == 0:
                            gau_label.append(1)
                        else:
                            gau_label.append(0)

                    axes[i][j].scatter(
                        show_data[:, 0],
                        show_data[:, 1],
                        c=gau_label,
                        # cmap="RdYlGn",
                        s=7,
                        marker=".",
                    )
                else:
                    axes[i][j].scatter(
                        show_data[:, 0],
                        show_data[:, 1],
                        c=label,
                        # cmap="RdYlGn",
                        s=7,
                        marker=".",
                    )

                axes[i][j].set_xticks([])
                axes[i][j].set_yticks([])

                if i == 5:
                    axes[5][j].set_xlabel(
                        self.cicrles_title_list[j], fontdict=self.font
                    )
                if j == 0:
                    axes[i][0].set_ylabel(
                        # "Noise level = "
                        str("{:.2f}".format(self.params["circles"]["noise"][i]))
                        + "        ",
                        rotation=0,
                        fontdict=self.font,
                    )

        plt.tight_layout()
        plt.savefig(
            self.save_path
            + "circles/num_"
            + str(self.params["circles"]["num"])
            + ".pdf",
            bbox_inches="tight",
        )
        plt.show()


def _show_data_algorithm(plot, path, data_name, algorithm_name, pred_result_file=""):
    """
    绘制一个数据集上一个算法的聚类结果图
    Args:
        plot (_type_): 绘图句柄
        path (_type_): 工作目录
        data_name (_type_): 数据集名称
        algorithm_list (_type_): 算法名称
        pred_result_file (str): 预测结果文件。如果不指定，从 analyze 下获取；否则从 result 下加 pred_result_file 得到
    """
    """原始数据"""
    data = pandas.read_csv(path + "analyze/" + data_name + ".csv")

    """寻找 algorithm_name 对应的文件"""
    file_name = ""
    for file in os.listdir(path + "analyze/" + data_name):
        if file.find(algorithm_name + "+") == 0:
            file_name = file

    """预测结果"""
    if pred_result_file == "":
        """从 analyze 下的 file_name 获取"""
        with open(path + "analyze/" + data_name + "/" + file_name, "r") as f:
            pred_result = json.load(f)
    else:
        """从 result 下的 pred_result_file 获取"""
        with open(
            path
            + "result/"
            + data_name
            + "/"
            + algorithm_name
            + "/"
            + pred_result_file,
            "r",
        ) as f:
            pred_result = json.load(f)

    label = pred_result["label"]
    """将预测的标签写入到原始数据中"""
    data["label"] = label
    """列标签"""
    col = list(data.columns)

    """收集样本"""
    cluster_result = {k: list() for k in set(label)}
    for i in range(len(label)):
        cluster_result[label[i]].append(i)

    """聚类结果字典"""
    cluster_points = dict()
    colors = dict()
    numpy.random.seed(1)

    for k, v in cluster_result.items():
        """同一类中的点"""
        cluster_points[k] = data.loc[cluster_result[k], :]
        colors[k] = numpy.random.rand(3).reshape(1, -1)

    """找出聚类中心"""
    center = list()
    if algorithm_name.find("dpc") != -1:
        center = [data.loc[idx, :] for idx in pred_result["center"]]
    elif algorithm_name.find("kmeans") != -1:
        center = pred_result["center"]
    center = numpy.array(center)

    """绘图"""
    idx = 0
    for k, v in cluster_points.items():
        plot.scatter(v.loc[:, col[0]], v.loc[:, col[1]], c=colors[k], s=4, marker=".")
        if len(center) != 0:
            plot.scatter(center[idx, 0], center[idx, 1], c=colors[k], s=256, marker="*")
            idx += 1

    """图像设置"""
    font = {
        "family": "Times New Roman",
    }
    plot.spines["right"].set_color("none")
    plot.spines["top"].set_color("none")
    plot.set_xlim(0, 1.03)
    plot.set_ylim(0, 1.03)
    plot.set_xticks(numpy.linspace(0, 1, 11))
    plot.set_yticks(numpy.linspace(0, 1, 11))
    plot.set_aspect(1)
    # plot.grid(linestyle="-", linewidth=0.3)
    plot.set_xlabel("X")
    plot.set_ylabel("Y")
    if algorithm_name in PLOT_TITLE.keys():
        plot.set_title(PLOT_TITLE[algorithm_name], font)


def _show_data(plot, path, data_name):
    """
    绘制一个数据集的原始分布
    Args:
        plot (_type_): 绘图句柄
        path (_type_): 工作目录
        data_name (_type_): 数据集名称
    Returns:
    """
    """原始数据"""
    data = pandas.read_csv(path + "analyze/" + data_name + ".csv")
    col = list(data.columns)
    """分离原始数据与原始标签"""
    origin_data = data.loc[:, col[0:-1]]
    origin_label = data[col[-1]]

    """收集样本"""
    cluster_result = {k: list() for k in set(origin_label)}
    for i in range(len(origin_label)):
        cluster_result[origin_label[i]].append(i)

    """聚类结果字典"""
    cluster_points = dict()
    colors = dict()
    numpy.random.seed(6)

    for k, v in cluster_result.items():
        """同一类中的点"""
        cluster_points[k] = data.loc[cluster_result[k], :]
        colors[k] = numpy.random.rand(3).reshape(1, -1)

    """绘图"""
    idx = 0
    for k, v in cluster_points.items():
        plot.scatter(v.loc[:, col[0]], v.loc[:, col[1]], c=colors[k], s=15, marker=".")
        idx += 1

    """图像设置"""
    font = {
        "family": "Times New Roman",
    }

    plot.spines["right"].set_color("none")
    plot.spines["top"].set_color("none")
    plot.set_xlim(0, 1.03)
    plot.set_ylim(0, 1.03)
    plot.set_xticks(numpy.linspace(0, 1, 11))
    plot.set_yticks(numpy.linspace(0, 1, 11))
    plot.set_aspect(1)
    plot.grid(linestyle="-", linewidth=0.5)
    plot.set_xlabel("X")
    plot.set_ylabel("Y")
    plot.set_title(data_name, font)


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
            dpc_algorithm (str, optional): 对比算法. Defaults to "dpc_irho".
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
            al (_type_): 算法对象
            file_name (_type_): 文件名
        """
        """遍历文件夹，获取文件名"""
        file_name = ""
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

        return al, file_name

    @DeprecationWarning
    def rho_cluster(self, axes, data_name, algorithm_name):
        """
        绘制一个算法在一个数据集上的局部密度图与聚类结果图(已经拆至 plot_rho)
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
        al, file_name = self.gain_cluster_rho(data_name, algorithm_name)
        rho = al.rho

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
                al_center[index], rho[al_center[index]], c=colors[k], s=256, marker="*"
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
                s=256,
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

    def plot_rho(self, plot, data_name, algorithm_name, plot_type):
        """
        绘制一个算法在一个数据集上的图，由 plot_type 指定
        Args:
            plot (_type_): 绘图句柄
            data_name (_type_): 文件名
            algorithm_name (_type_): 算法名称
            plot_type (_type_): 绘图类型(rho，delta，dot，gamma)，聚类结果从 _show_data_algorithm 获取
        """
        """原始数据"""
        data = pandas.read_csv(self.path + "analyze/" + data_name + ".csv")
        col = list(data.columns)
        """分离原始数据与原始标签"""
        origin_data = data.loc[:, col[0:-1]]

        """获取聚类结果参数"""
        al, file_name = self.gain_cluster_rho(data_name, algorithm_name)
        """获取局部密度，相对距离"""
        rho = al.rho
        delta = al.delta
        """决策值"""
        gamma = rho * delta

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
        numpy.random.seed(6)

        """根据聚类结果拆分数据，得到一个字典，键为类别，值为归属该类的样本索引"""
        al_cluster_result = dict()
        for k in set(al_label):
            colors[k] = numpy.random.rand(3).reshape(1, -1)
            al_cluster_result[k] = list()

        for index in range(len(al_label)):
            al_cluster_result[al_label[index]].append(index)

        index = 0
        for k, v in al_cluster_result.items():
            """图像设置"""
            plot.spines["right"].set_color("none")
            plot.spines["top"].set_color("none")
            self.font["size"] = 16

            """判断绘图类型"""
            if plot_type == "rho":
                """局部密度"""
                plot.scatter(v, rho[v], c=colors[k], s=15, marker=".")
                plot.scatter(
                    al_center[index],
                    rho[al_center[index]],
                    c=colors[k],
                    s=256,
                    marker="*",
                )
                """x 轴，点的个数"""
                plot.set_xlim(0, len(al_label) * 1.1)
                plot.set_ylim(0, max(rho) * 1.1)
                plot.set_xlabel("num")
                plot.set_ylabel(r"$\rho$")

                if algorithm_name == "dpc":
                    plot.set_title(r"$\rho$ value of DPC", self.font, y=-0.15)
                else:
                    plot.set_title(r"$\rho$ value of LW-DPC", self.font, y=-0.15)

            elif plot_type == "delta":
                """相对距离"""
                plot.scatter(v, delta[v], c=colors[k], s=15, marker=".")
                plot.scatter(
                    al_center[index],
                    delta[al_center[index]],
                    c=colors[k],
                    s=256,
                    marker="*",
                )
                plot.set_xlim(0, len(al_label) * 1.1)
                plot.set_ylim(0, max(delta) * 1.1)
                plot.set_xlabel("num")
                plot.set_ylabel(r"$\delta$")

                if algorithm_name == "dpc":
                    plot.set_title(r"$\delta$ value of DPC", self.font, y=-0.15)
                else:
                    plot.set_title(r"$\delta$ value of LW-DPC", self.font, y=-0.15)

            elif plot_type == "dot":
                """局部密度 x 相对距离，记作 dot"""
                plot.scatter(rho[v], delta[v], c=colors[k], s=15, marker=".")
                plot.scatter(
                    rho[al_center[index]],
                    delta[al_center[index]],
                    c=colors[k],
                    s=256,
                    marker="*",
                )
                plot.set_xlim(0, max(rho) * 1.1)
                plot.set_ylim(0, max(delta) * 1.1)
                plot.set_xlabel(r"$\rho$")
                plot.set_ylabel(r"$\delta$")

                if algorithm_name == "dpc":
                    plot.set_title(
                        r"$\rho$ and $\delta$ value of DPC", self.font, y=-0.15
                    )
                else:
                    plot.set_title(
                        r"$\rho$ and $\delta$ value of LW-DPC", self.font, y=-0.15
                    )

            elif plot_type == "gamma":
                """决策值"""
                plot.scatter(v, gamma[v], c=colors[k], s=15, marker=".")
                plot.scatter(
                    al_center[index],
                    gamma[al_center[index]],
                    c=colors[k],
                    s=256,
                    marker="*",
                )
                plot.set_xlim(0, len(al_label) * 1.1)
                plot.set_ylim(0, max(gamma) * 1.1)
                plot.set_xlabel("num")
                plot.set_ylabel(r"$\gamma$")

                if algorithm_name == "dpc":
                    plot.set_title(r"$\gamma$ value of DPC", self.font, y=-0.15)
                else:
                    plot.set_title(r"$\gamma$ value of LW-DPC", self.font, y=-0.15)

            plot.grid(linestyle="-", linewidth=0.5)
            index += 1

    def show_dpc_process(self):
        """
        展示 DPC 聚类算法的过程，2 x 3，对应图 2-1
        (原始数据，局部密度，相对聚类
        局部密度 x 相对距离，决策值，聚类结果)
        数据集暂定为 Aggregation
        """
        """数据集名称"""
        data_name = "aggregation"

        """2 x 3 的格式"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        """(0,0)，原始数据"""
        _show_data(axes[0][0], self.path, data_name)
        axes[0][0].set_title("(a) " + data_name.capitalize(), self.font, y=-0.15)
        """(0,1)，局部密度"""
        self.plot_rho(axes[0][1], data_name, "dpc", "rho")
        axes[0][1].set_title(r"(b) $\rho$ value of DPC", self.font, y=-0.15)
        """(0,2)，相对距离"""
        self.plot_rho(axes[0][2], data_name, "dpc", "delta")
        axes[0][2].set_title(r"(c) $\delta$ value of DPC", self.font, y=-0.15)
        """(1,0)，局部密度与相对距离"""
        self.plot_rho(axes[1][0], data_name, "dpc", "dot")
        axes[1][0].set_title(r"(d) $\rho$ $\times$ $\delta$ of DPC", self.font, y=-0.15)
        """(1,1)，gamma"""
        self.plot_rho(axes[1][1], data_name, "dpc", "gamma")
        axes[1][1].set_title(r"(e) $\gamma$ value of DPC", self.font, y=-0.15)
        """(1,2)，聚类结果"""
        _show_data_algorithm(axes[1][2], self.path, data_name, "dpc")
        axes[1][2].set_title(r"(f) Clustering result of DPC", self.font, y=-0.15)

        """保存图片"""
        plt.tight_layout()
        plt.savefig(
            self.path + "plot/DPC在" + data_name + "上的聚类过程.pdf",
            bbox_inches="tight",
        )
        plt.show()

    def show_dpc_compare(self):
        """
        DPC 对比其他算法 2 x 2，对应图 2-2
        数据集暂定为 spiral
        """
        """数据集名称"""
        data_name = "spiral"

        """2 x 2 的格式"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        tmp_algorithm_list = ["kmeans", "optics", "spectralClustering", "dpc"]
        for i in range(2):
            for j in range(2):
                """绘图"""
                _show_data_algorithm(
                    axes[i][j], self.path, data_name, tmp_algorithm_list[i * 2 + j]
                )
                axes[i][j].set_title(
                    PLOT_TITLE_NUM[i * 2 + j]
                    + PLOT_TITLE[tmp_algorithm_list[i * 2 + j]],
                    self.font,
                    y=-0.15,
                )

        """保存图片"""
        plt.tight_layout()
        plt.savefig(self.path + "plot/DPC_" + data_name + ".pdf", bbox_inches="tight")
        plt.show()

    def show_distance_compare(self):
        """
        不同样本相似性度量对 DPC 的聚类结果的影响(影响大与不大的都放一张图里面)，2 x 3，对应图 2-3
        数据集暂定为 D31
        """
        data_name = "D31"
        distance_list = [
            "chebyshev",
            "cityblock",
            "cosine",
            "euclidean",
            "jaccard",
            "mahalanobis",
        ]

        """2 x 3 的格式"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 13))

        for i in range(2):
            for j in range(3):
                """绘图"""
                _show_data_algorithm(
                    axes[i][j], self.path, data_name, "dpc_" + distance_list[i * 3 + j]
                )
                axes[i][j].set_title(
                    PLOT_TITLE_NUM[i * 3 + j]
                    + " DPC with "
                    + distance_list[i * 3 + j].capitalize(),
                    self.font,
                    y=-0.15,
                )

        """保存图片"""
        plt.tight_layout()
        plt.savefig(
            self.path + "plot/DPC_distance_" + data_name + ".pdf", bbox_inches="tight"
        )
        plt.show()

    def show_percent_compare(self):
        """
        不同百分比数对 DPC 的聚类结果的影响，2 x 3，对应图 2-4
        数据集暂定为 Flame
        """
        data_name = "flame"
        percent_list = [0.5, 1.5, 2.5, 3.0]

        """2 x 2 的格式"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 13))

        for i in range(2):
            for j in range(2):
                """绘图"""
                _show_data_algorithm(
                    axes[i][j],
                    self.path,
                    data_name,
                    "dpc",
                    "dcp_"
                    + str(percent_list[i * 2 + j])
                    + "__dcm_1__rhom_0__dem_0.json",
                )
                axes[i][j].set_title(
                    PLOT_TITLE_NUM[i * 2 + j]
                    + " DPC with percent = "
                    + str(percent_list[i * 2 + j]),
                    self.font,
                    y=-0.15,
                )

        """保存图片"""
        plt.tight_layout()
        plt.savefig(
            self.path + "plot/DPC_percent_" + data_name + ".pdf", bbox_inches="tight"
        )
        plt.show()

    def show_rho_compare(self):
        """
        使用的数据集为 jain
        改进局部密度的 DPC 算法与经典 DPC 算法在局部密度与聚类结果的对比图，2 x 2，对应图 3-2
        """
        data_name = "jain"
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        self.plot_rho(axes[0][0], data_name, "dpc", "rho")
        axes[0][0].set_title(r"(a) $\rho$ value of DPC", self.font, y=-0.15)
        _show_data_algorithm(axes[0][1], self.path, data_name, "dpc")
        axes[0][1].set_title(r"(b) Clustering result of DPC", self.font, y=-0.15)

        self.plot_rho(axes[1][0], data_name, "dpc_irho", "rho")
        axes[1][0].set_title(r"(c) $\rho$ value of LW-DPC", self.font, y=-0.15)
        _show_data_algorithm(axes[1][1], self.path, data_name, "dpc_irho")
        axes[1][1].set_title(r"(d) Clustering result of LW-DPC", self.font, y=-0.15)

        """保存图片"""
        plt.tight_layout()
        plt.savefig(self.path + "plot/改进局部密度对比.pdf", bbox_inches="tight")
        plt.show()

    def show_lw_dpc_defect(self):
        """
        展示 LW-DPC 的不足，即选择了恰当的聚类中心，聚类结果也一般，2 x 2，对应图 4-1
        选定的数据集为 D31
        """
        """数据集，算法为 DPC，KNN-DPC，SNN-DPC，LW-DPC"""
        data_name = "D31"

        fig, axes = plt.subplots(2, 2, figsize=(18, 18))
        acc_list = ["0.9690", "0.9710", "0.9761", "0.9706"]

        for i in range(2):
            for j in range(2):
                """绘图"""
                _show_data_algorithm(
                    axes[i][j], self.path, data_name, self.algorithm_list[5 + i * 2 + j]
                )
                axes[i][j].set_title(
                    PLOT_TITLE_NUM[i * 2 + j]
                    + PLOT_TITLE[self.algorithm_list[5 + 2 * i + j]]
                    + "(ACC = "
                    + acc_list[i * 2 + j]
                    + ")",
                    self.font,
                    y=-0.1,
                )

        """保存图片"""
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0.1)
        plt.savefig(self.path + "plot/lw_dpc_deffect.pdf")
        plt.show()

    def show_cluster_results(self):
        """
        展示 Synthesis 数据集上每个聚类算法的聚类结果，通解
        """
        """不同数据集"""
        for data_name in self.dataset_list:
            """3 x 3 的格式"""
            fig, axes = plt.subplots(3, 3, figsize=(18, 18))
            for i in range(3):
                for j in range(3):
                    """绘图"""
                    _show_data_algorithm(
                        axes[i][j], self.path, data_name, self.algorithm_list[i * 3 + j]
                    )

            """保存图片"""
            plt.tight_layout()
            plt.savefig(self.path + "plot/" + data_name + ".svg", bbox_inches="tight")
            plt.show()


class PlotUci:
    """
    处理 Uci 数据集相关的绘图
    1. 根据 ./result/uci/analyze/ 下的数据集绘制图，将图存放到 ./result/uci/plot/，需要降维
    2. 参数验证(也涉及到 Synthesis 部分的数据集)，将图统一存放到 ./result/uci/plot/，默认是四个聚类指标都有，ACC，AMI，ARI，FMI
    """

    def __init__(self, path="./result/", dpc_algorithm=["dpc_iass"]) -> None:
        """
        初始化相关成员
        Args:
            path (str, optional): 工作目录. Defaults to "./result/".
            dpc_algorithm (str, optional): 对比算法. Defaults to "dpc_iass".
        """
        """保存结果路径"""
        self.path = path
        """字体配置"""
        self.font = {
            "family": "Times New Roman",
            # "color": "black",
            "size": 16,
        }
        """合成数据集"""
        self.synthesis_dataset_list = [
            "aggregation",
            "compound",
            "D31",
            "flame",
            "jain",
            "pathbased",
            "R15",
            "S2",
            "spiral",
        ]
        """UCI 数据集"""
        self.param_order_dataset_list = [
            "abalone",
            "blood",
            "dermatology",
            "ecoli",
            "glass",
            "iris",
            "isolet",
            "jaffe",
            "letter",
            "libras",
            "lung",
            "magic",
            "parkinsons",
            "pima",
            "seeds",
            "segment",
            "sonar",
            "spambase",
            "teaching",
            "tox171",
            "twonorm",
            "usps",
            "waveform",
            "waveformNoise",
            "wdbc",
            "wilt",
            "wine",
        ]
        """添加对比算法"""
        self.dpc_algorithm = dpc_algorithm

    def gain_cluster_index(self, dataset_list, param_dict):
        """
        寻找对应参数下算法在 dataset_list 上每个数据集的聚类指标
        Args:
            dataset_list (_type_): 数据集列表
            param_dict (_type_): 参数字典，依据参数寻找对应的文件，并获得聚类指标
            以单一数据集角度，用到哪种传哪种：
            {
                "dataset":{
                    "euc_mu": -, 寻找的是欧式距离下所有的 k 的文件对应的指标列表
                    "ckrod_mu": mu_value, 寻找的是该 mu_value 下所有的 k 的文件对应的指标列表
                    "ckrod_k": k_value, 寻找的是该 k_value 下所有的 mu 的文件对应的指标列表
                    "order_sensitivity": {"k": k_value, "mu": mu_value}, 寻找的是该列表内所有的样本顺序的文件对应的指标列表
                }
            }
        Returns:
            acc_dict (_type_): acc
            ami_dict (_type_): ami
            ari_dict (_type_): ari
            fmi_dict (_type_): fmi
        """
        """指标字典"""
        acc_dict = dict()
        ami_dict = dict()
        ari_dict = dict()
        fmi_dict = dict()

        for data in dataset_list:
            """文件路径"""
            data_path = self.path
            """数据集对应的参数"""
            data_param = param_dict[data]
            """指标"""
            acc_dict[data] = list()
            ami_dict[data] = list()
            ari_dict[data] = list()
            fmi_dict[data] = list()

            if data in self.synthesis_dataset_list:
                """合成数据集"""
                data_path += "synthesis/"
            else:
                """uci 数据集"""
                # data_path += "uci/"
                data_path += "real/"

            """添加数据集与算法路径"""
            data_path += "result/" + data + "/"

            """根据参数类型判断寻找什么样的文件"""
            if "euc_mu" in data_param.keys():
                """寻找的是欧式距离下所有的 k 的文件对应的指标列表"""
                data_path += self.dpc_algorithm[0] + "/"
                """k 的范围是 [4, 64]"""
                file_list = [
                    "rho_3__dem_1__ush_0__k_" + str(k) + ".json"
                    for k in list(range(4, 65))
                ]
            elif "ckrod_mu" in data_param.keys():
                """寻找的是该 mu_value 下所有的 k 的文件对应的指标列表"""
                data_path += self.dpc_algorithm[1] + "/"
                """mu 确定，k 的范围是 [4, 64]"""
                file_list = [
                    "rho_3__dem_1__ush_0__k_"
                    + str(k)
                    + "__mu_"
                    + str(float(data_param["ckrod_mu"]))
                    + ".json"
                    for k in list(range(4, 65))
                ]
            elif "ckrod_k" in data_param.keys():
                """寻找的是该 k_value 下所有的 mu 的文件对应的指标列表"""
                data_path += self.dpc_algorithm[1] + "/"
                """k 确定，mu 的范围是 [1, 100]"""
                file_list = [
                    "rho_3__dem_1__ush_0__k_"
                    + str(int(data_param["ckrod_k"]))
                    + "__mu_"
                    + str(float(mu))
                    + ".json"
                    for mu in list(range(1, 101))
                ]
            elif "order_sensitivity" in data_param.keys():
                """寻找的是该列表内所有的样本顺序的文件对应的指标列表"""
                if "euc_mu" in data_param["order_sensitivity"]:
                    """只有 k"""
                    data_path += self.dpc_algorithm[0] + "/"
                    file_list = [
                        "k_"
                        + str(data_param["order_sensitivity"]["euc_mu"])
                        + "__idx_"
                        + str(idx)
                        + ".json"
                        for idx in range(1, 31)
                    ]
                else:
                    """有 k 跟 mu"""
                    data_path += self.dpc_algorithm[1] + "/"
                    file_list = [
                        "k_"
                        + str(data_param["order_sensitivity"]["ckrod_k"])
                        + "__idx_"
                        + str(idx)
                        + "__mu_"
                        + str(data_param["order_sensitivity"]["ckrod_mu"])
                        + ".json"
                        for idx in range(1, 31)
                    ]

            """读取文件"""
            for file in file_list:
                with open(data_path + file, "r") as f:
                    d = json.load(f)

                acc_dict[data].append(round(d["cluster_acc"], 4))
                ami_dict[data].append(round(d["adjusted_mutual_info"], 4))
                ari_dict[data].append(round(d["adjusted_rand_index"], 4))
                fmi_dict[data].append(round(d["fowlkes_mallows_index"], 4))

        return acc_dict, ami_dict, ari_dict, fmi_dict

    def _show_cluster_index(self, plot, cluster_index, dataset_list):
        """
        绘制一个聚类指标
        Args:
            plot (_type_): 绘图句柄
            cluster_index (_type_): 聚类指标
            dataset_list (_type_): 涉及的数据集，主要用来画图例
        """
        """指标结果长度判断"""
        x_len = len(cluster_index[dataset_list[1]])
        if x_len == 61:
            """k"""
            xticks = list(range(4, 65))
            plot.set_xticks(numpy.linspace(xticks[0], xticks[-1], 13))
            plot.set_xlim(xticks[0], xticks[-1])
        elif x_len == 100:
            """mu"""
            xticks = list(range(1, 101))
            plot.set_xticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            plot.set_xlim(xticks[0], xticks[-1])
        elif x_len == 30:
            """random state"""
            xticks = list(range(1, 31))
            plot.set_xticks([1, 5, 10, 15, 20, 25, 30])
            plot.set_xlim(xticks[0], xticks[-1])

        for index_list in cluster_index.values():
            plot.plot(xticks, index_list, marker=".", linewidth=1, markersize=8)

        plot.set_ylim(0, 1.05)
        plot.set_yticks(numpy.linspace(0, 1, 11))
        plot.legend(dataset_list, loc="lower right", prop=self.font)

    def show_euc_k(self):
        """
        对使用欧式距离作为度量的 DPC-TAS 的 k 进行讨论，对应第四章，倒数第四张图
        """
        """使用的数据集"""
        dataset_list = ["aggregation", "D31", "flame", "R15", "S2", "spiral"]
        """每个数据集对应的参数"""
        param_dict = {
            "aggregation": {"euc_mu": 0},
            "D31": {"euc_mu": 0},
            "flame": {"euc_mu": 0},
            "R15": {"euc_mu": 0},
            "S2": {"euc_mu": 0},
            "spiral": {"euc_mu": 0},
            "jain": {"euc_mu": 0},
        }
        """指标字典"""
        acc_dict, ami_dict, ari_dict, fmi_dict = self.gain_cluster_index(
            dataset_list, param_dict
        )

        """绘图"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 18))
        self._show_cluster_index(axes[0][0], acc_dict, dataset_list)
        axes[0][0].set_title(PLOT_TITLE_NUM[0] + "ACC", self.font, y=-0.12)
        axes[0][0].set_xlabel(r"k", fontdict=self.font)
        axes[0][0].set_ylabel(r"ACC", fontdict=self.font)
        self._show_cluster_index(axes[0][1], ami_dict, dataset_list)
        axes[0][1].set_title(PLOT_TITLE_NUM[1] + "AMI", self.font, y=-0.12)
        axes[0][1].set_xlabel(r"k", fontdict=self.font)
        axes[0][1].set_ylabel(r"AMI", fontdict=self.font)
        self._show_cluster_index(axes[1][0], ari_dict, dataset_list)
        axes[1][0].set_title(PLOT_TITLE_NUM[2] + "ARI", self.font, y=-0.12)
        axes[1][0].set_xlabel(r"k", fontdict=self.font)
        axes[1][0].set_ylabel(r"ARI", fontdict=self.font)
        self._show_cluster_index(axes[1][1], fmi_dict, dataset_list)
        axes[1][1].set_title(PLOT_TITLE_NUM[3] + "FMI", self.font, y=-0.12)
        axes[1][1].set_xlabel(r"k", fontdict=self.font)
        axes[1][1].set_ylabel(r"FMI", fontdict=self.font)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.18)
        plt.savefig("./result/uci/plot/euc_k.pdf", bbox_inches="tight")
        plt.show()

    def show_ckrod_k(self):
        """
        对使用 CKROD 作为度量的 DPC-TAS 的 k 进行讨论，对应第四章，倒数第三张图
        """
        """使用的数据集"""
        dataset_list = ["Dermatology", "Iris", "Seeds", "Wine"]
        """每个数据集对应的参数"""
        param_dict = {
            "Dermatology": {"ckrod_mu": 9},
            "Iris": {"ckrod_mu": 20},
            "Seeds": {"ckrod_mu": 12},
            "Wine": {"ckrod_mu": 1},
        }
        """指标字典"""
        acc_dict, ami_dict, ari_dict, fmi_dict = self.gain_cluster_index(
            dataset_list, param_dict
        )

        """绘图"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 18))
        self._show_cluster_index(axes[0][0], acc_dict, dataset_list)
        axes[0][0].set_title(PLOT_TITLE_NUM[0] + "ACC", self.font, y=-0.12)
        axes[0][0].set_xlabel(r"k", fontdict=self.font)
        axes[0][0].set_ylabel(r"ACC", fontdict=self.font)
        self._show_cluster_index(axes[0][1], ami_dict, dataset_list)
        axes[0][1].set_title(PLOT_TITLE_NUM[1] + "AMI", self.font, y=-0.12)
        axes[0][1].set_xlabel(r"k", fontdict=self.font)
        axes[0][1].set_ylabel(r"AMI", fontdict=self.font)
        self._show_cluster_index(axes[1][0], ari_dict, dataset_list)
        axes[1][0].set_title(PLOT_TITLE_NUM[2] + "ARI", self.font, y=-0.12)
        axes[1][0].set_xlabel(r"k", fontdict=self.font)
        axes[1][0].set_ylabel(r"ARI", fontdict=self.font)
        self._show_cluster_index(axes[1][1], fmi_dict, dataset_list)
        axes[1][1].set_title(PLOT_TITLE_NUM[3] + "FMI", self.font, y=-0.12)
        axes[1][1].set_xlabel(r"k", fontdict=self.font)
        axes[1][1].set_ylabel(r"FMI", fontdict=self.font)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.18)
        plt.savefig("./result/uci/plot/ckrod_k.pdf", bbox_inches="tight")
        plt.show()

    def show_ckrod_mu(self):
        """
        对使用 CKROD 作为度量的 DPC-TAS 的 mu 进行讨论，对应第四章，倒数第二张图
        """
        """使用的数据集"""
        dataset_list = ["Dermatology", "Iris", "Seeds", "Wine"]
        """每个数据集对应的参数"""
        param_dict = {
            "Dermatology": {"ckrod_k": 16},
            "Iris": {"ckrod_k": 17},
            "Seeds": {"ckrod_k": 7},
            "Wine": {"ckrod_k": 27},
        }
        """指标字典"""
        acc_dict, ami_dict, ari_dict, fmi_dict = self.gain_cluster_index(
            dataset_list, param_dict
        )

        """绘图"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 18))
        self._show_cluster_index(axes[0][0], acc_dict, dataset_list)
        axes[0][0].set_title(PLOT_TITLE_NUM[0] + "ACC", self.font, y=-0.12)
        axes[0][0].set_xlabel(r"$\mu$", fontdict=self.font)
        axes[0][0].set_ylabel(r"ACC", fontdict=self.font)
        self._show_cluster_index(axes[0][1], ami_dict, dataset_list)
        axes[0][1].set_title(PLOT_TITLE_NUM[1] + "AMI", self.font, y=-0.12)
        axes[0][1].set_xlabel(r"$\mu$", fontdict=self.font)
        axes[0][1].set_ylabel(r"AMI", fontdict=self.font)
        self._show_cluster_index(axes[1][0], ari_dict, dataset_list)
        axes[1][0].set_title(PLOT_TITLE_NUM[2] + "ARI", self.font, y=-0.12)
        axes[1][0].set_xlabel(r"$\mu$", fontdict=self.font)
        axes[1][0].set_ylabel(r"ARI", fontdict=self.font)
        self._show_cluster_index(axes[1][1], fmi_dict, dataset_list)
        axes[1][1].set_title(PLOT_TITLE_NUM[3] + "FMI", self.font, y=-0.12)
        axes[1][1].set_xlabel(r"$\mu$", fontdict=self.font)
        axes[1][1].set_ylabel(r"FMI", fontdict=self.font)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.18)
        plt.savefig("./result/uci/plot/ckrod_mu.pdf", bbox_inches="tight")
        plt.show()

    def show_order_sensitivity(self):
        """
        对使用 CKROD 作为度量的 DPC-TAS 的样本顺序进行讨论，对应第四章，倒数第一张图
        """
        """使用的数据集"""
        dataset_list = [
            "spiral",
            "R15",
            "D31",
            "Seeds",
            "Isolet",
            "Parkinsons",
            "Segment",
        ]
        """每个数据集对应的参数"""
        param_dict = {
            "spiral": {"order_sensitivity": {"euc_mu": 4}},
            "R15": {"order_sensitivity": {"euc_mu": 10}},
            "D31": {"order_sensitivity": {"euc_mu": 27}},
            "Seeds": {"order_sensitivity": {"ckrod_k": 7, "ckrod_mu": 1.0}},
            "Isolet": {"order_sensitivity": {"ckrod_k": 23, "ckrod_mu": 20.0}},
            "Parkinsons": {"order_sensitivity": {"ckrod_k": 8, "ckrod_mu": 15.0}},
            "Segment": {"order_sensitivity": {"ckrod_k": 18, "ckrod_mu": 11.0}},
        }
        """指标字典"""
        acc_dict, ami_dict, ari_dict, fmi_dict = self.gain_cluster_index(
            dataset_list, param_dict
        )

        """绘图"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 18))
        self._show_cluster_index(axes[0][0], acc_dict, dataset_list)
        axes[0][0].set_title(PLOT_TITLE_NUM[0] + "ACC", self.font, y=-0.12)
        axes[0][0].set_xlabel(r"random state", fontdict=self.font)
        axes[0][0].set_ylabel(r"ACC", fontdict=self.font)
        self._show_cluster_index(axes[0][1], ami_dict, dataset_list)
        axes[0][1].set_title(PLOT_TITLE_NUM[1] + "AMI", self.font, y=-0.12)
        axes[0][1].set_xlabel(r"random state", fontdict=self.font)
        axes[0][1].set_ylabel(r"AMI", fontdict=self.font)
        self._show_cluster_index(axes[1][0], ari_dict, dataset_list)
        axes[1][0].set_title(PLOT_TITLE_NUM[2] + "ARI", self.font, y=-0.12)
        axes[1][0].set_xlabel(r"random state", fontdict=self.font)
        axes[1][0].set_ylabel(r"ARI", fontdict=self.font)
        self._show_cluster_index(axes[1][1], fmi_dict, dataset_list)
        axes[1][1].set_title(PLOT_TITLE_NUM[3] + "FMI", self.font, y=-0.12)
        axes[1][1].set_xlabel(r"random state", fontdict=self.font)
        axes[1][1].set_ylabel(r"FMI", fontdict=self.font)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.18)
        plt.savefig("./result/uci/plot/sample_order.pdf", bbox_inches="tight")
        plt.show()


class PlotImage:
    """
    处理 Image 数据集相关的绘图
    主要是聚类结果展示，由于图像数据集相对不多，且每个都需要展示，因此每个数据集一个函数方法
    与之前相对不一样的是，需要根据得到的文件去获取标签以及原始数据，绘制部分结果
    第一列是聚类中心对应的样本，其余列是部分结果
    行数是数据集的标签类别数
    """

    def __init__(self, save_path="./result/image/", dpc_algorithm="dpc_iass") -> None:
        """
        初始化相关成员
        Args:
            save_path (str, optional): 保存结果的路径. Defaults to "./result/image/".
            dpc_algorithm (str, optional): 对比算法. Defaults to "dpc_iass".
        """
        """保存结果路径"""
        self.save_path = save_path
        """字体配置"""
        self.font = {
            "family": "Times New Roman",
            "color": "black",
            "size": 16,
        }
        """对比算法"""
        self.dpc_algorithm = dpc_algorithm

    def _show_image(self, data_name, file_name, col_num, row_num):
        """
        绘制一个数据集在特定参数下的多子像素图
        Args:
            data_name (_type_): _description_
            file_name (_type_): _description_
            col_num (_type_): _description_
            row_num (_type_): _description_
        """
        """行数"""
        row_num = min(IMAGE_PARAMS[data_name]["num"], row_num)
        """特征个数"""
        feature_num = IMAGE_PARAMS[data_name]["features_num"]
        """根据文件加载聚类中心，标签"""
        with open(
            self.save_path
            + "result/"
            + data_name
            + "/"
            + self.dpc_algorithm
            + "/"
            + file_name
        ) as f:
            d = json.load(f)
        """加载原始数据"""
        raw_data = loadmat(self.save_path + "analyze/" + data_name + ".mat")["X"]

        """获取预测标签，预测聚类中心"""
        pred_label = d["label"]
        pred_center = d["center"]
        """转化为字典"""
        pred_dict = {c: [] for c in pred_center}
        for idx in range(len(pred_label)):
            for c in pred_center:
                if pred_label[c] == pred_label[idx]:
                    pred_dict[c].append(idx)

        fig, axes = plt.subplots(row_num, col_num, figsize=(col_num, row_num))

        for i in range(row_num):
            for j in range(col_num):
                """绘制聚类中心"""
                if j == 0:
                    axes[i][j].imshow(
                        numpy.rot90(
                            raw_data[pred_center[i]].reshape(
                                int(math.sqrt(feature_num)), int(math.sqrt(feature_num))
                            ),
                            k=3,
                        )
                    )
                else:
                    axes[i][j].imshow(
                        numpy.rot90(
                            raw_data[pred_dict[pred_center[i]][j]].reshape(
                                int(math.sqrt(feature_num)), int(math.sqrt(feature_num))
                            ),
                            k=3,
                        )
                    )
                axes[i][j].set_xticks(())
                axes[i][j].set_yticks(())

        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig("./result/image/plot/" + data_name + ".pdf", bbox_inches="tight")
        plt.show()

    def _show_mnist_test(self, data_name, file_name, col_num, row_num):
        """
        绘制一个数据集在特定参数下的多子像素图
        Args:
            data_name (_type_): _description_
            file_name (_type_): _description_
            col_num (_type_): _description_
            row_num (_type_): _description_
        """
        """行数"""
        row_num = min(IMAGE_PARAMS[data_name]["num"], row_num)
        """特征个数"""
        feature_num = IMAGE_PARAMS[data_name]["features_num"]
        """加载原始数据"""
        data = loadmat(self.save_path + "analyze/" + data_name + ".mat")
        """获取数据，标签"""
        raw_data = data["X"]
        raw_label = list(data["Y"][:, 0])
        list_set_label = list(set(raw_label))
        center_dict = {c: [] for c in list_set_label}
        for idx in range(len(raw_label)):
            for c in list_set_label:
                if c == raw_label[idx]:
                    center_dict[c].append(idx)
        """"""
        plot_center_dict = {c: [] for c in list_set_label}
        for k, v in center_dict.items():
            plot_center_dict[k] = random.sample(v, col_num)

        fig, axes = plt.subplots(row_num, col_num, figsize=(col_num, row_num))

        for i in range(row_num):
            for j in range(col_num):
                if j == 0:
                    """聚类中心"""
                    axes[i][j].imshow(
                        numpy.rot90(
                            raw_data[plot_center_dict[list_set_label[i]][j]].reshape(
                                int(math.sqrt(feature_num)), int(math.sqrt(feature_num))
                            ),
                            k=4,
                        ),
                        cmap="gray",
                    )
                else:
                    """其他样本"""
                    axes[i][j].imshow(
                        numpy.rot90(
                            raw_data[plot_center_dict[list_set_label[i]][j]].reshape(
                                int(math.sqrt(feature_num)), int(math.sqrt(feature_num))
                            ),
                            k=4,
                        ),
                        cmap="gray",
                    )

                axes[i][j].set_xticks(())
                axes[i][j].set_yticks(())

        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig("./result/image/plot/" + data_name + ".pdf", bbox_inches="tight")
        plt.show()

    def _show_usps(self, data_name, file_name, col_num, row_num):
        """
        绘制一个数据集在特定参数下的多子像素图
        Args:
            data_name (_type_): _description_
            file_name (_type_): _description_
            col_num (_type_): _description_
            row_num (_type_): _description_
        """
        """行数"""
        row_num = min(IMAGE_PARAMS[data_name]["num"], row_num)
        """特征个数"""
        feature_num = IMAGE_PARAMS[data_name]["features_num"]
        """加载原始数据"""
        data = loadmat(self.save_path + "analyze/" + data_name + ".mat")
        """获取数据，标签"""
        raw_data = data["X"]
        raw_label = list(data["Y"][:, 0])
        list_set_label = list(set(raw_label))
        center_dict = {c: [] for c in list_set_label}
        for idx in range(len(raw_label)):
            for c in list_set_label:
                if c == raw_label[idx]:
                    center_dict[c].append(idx)
        """"""
        plot_center_dict = {c: [] for c in list_set_label}
        for k, v in center_dict.items():
            plot_center_dict[k] = random.sample(v, col_num)
            print(k, plot_center_dict[k])
            
        fig, axes = plt.subplots(row_num, col_num, figsize=(col_num, row_num))

        for i in range(row_num):
            for j in range(col_num):
                if j == 0:
                    """聚类中心"""
                    axes[i][j].imshow(
                        numpy.rot90(
                            numpy.flipud(
                                raw_data[
                                    plot_center_dict[list_set_label[i]][j]
                                ].reshape(
                                    int(math.sqrt(feature_num)),
                                    int(math.sqrt(feature_num)),
                                ),
                            ),
                            k=-1,
                        ),
                        cmap="gray",
                    )
                else:
                    """其他样本"""
                    axes[i][j].imshow(
                        numpy.rot90(
                            numpy.flipud(
                                raw_data[
                                    plot_center_dict[list_set_label[i]][j]
                                ].reshape(
                                    int(math.sqrt(feature_num)),
                                    int(math.sqrt(feature_num)),
                                ),
                            ),
                            k=-1,
                        ),
                        cmap="gray",
                    )

                axes[i][j].set_xticks(())
                axes[i][j].set_yticks(())

        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig("./result/image/plot/" + data_name + ".pdf", bbox_inches="tight")
        plt.show()

    def show_coil20(self):
        """
        展示 coil20 数据集聚类结果，对应第三章的图
        """
        """coil20 数据集"""
        data_name = "coil20"
        """要画多少列，包括聚类中心"""
        col_num = 17
        """指定文件名"""
        file_name = "dem_1__k_34__mu_29.json"

        self._show_image(data_name, file_name, col_num, 10)

    def show_jaffe(self):
        """
        展示 jaffe 数据集聚类结果，对应第三章的图
        """
        """jaffe 数据集"""
        data_name = "jaffe"
        """要画多少列，包括聚类中心"""
        col_num = 16
        """指定文件名"""
        file_name = "dem_0__k_23__mu_30.json"

        self._show_image(data_name, file_name, col_num, IMAGE_PARAMS[data_name]["num"])

    def show_mnist_test(self):
        """
        展示 mnist_test 数据集聚类结果，对应第三章的图
        """
        """mnist_test 数据集"""
        data_name = "MNIST_test"
        """要画多少列，包括聚类中心"""
        col_num = 21
        """指定文件名"""
        file_name = ""

        self._show_mnist_test(
            data_name, file_name, col_num, IMAGE_PARAMS[data_name]["num"]
        )

    def show_usps(self):
        """
        展示 usps 数据集聚类结果，对应第三章的图
        """
        """usps 数据集"""
        data_name = "usps"
        """要画多少列，包括聚类中心"""
        col_num = 23
        """指定文件名"""
        file_name = ""

        self._show_usps(data_name, file_name, col_num, IMAGE_PARAMS[data_name]["num"])


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
            "size": 18,
        }
        self.config = {
            "font.family": "serif",
            "font.size": 12,
            "mathtext.fontset": "stix",
            "font.serif": ["SimSun"],
        }
        rcParams.update(self.config)

    def show_improve_rho(self):
        """
        对比局部密度的改进，对应图 3-4
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

    def show_domino_effect(self):
        """
        一步分配策略中的多米诺效应演示图，可以用现成数据集，但不用跟别人的重复，对应图 2-5
        """
        # plt.axis("off")
        ax = plt.gca()
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")

        plt.xlim(0, 1.0)
        plt.ylim(0, 1.0)

        ax.set_aspect(1)
        """cluster1"""
        plt.scatter(
            ONE_STEP_CLUSTERS["cluster1"][:, 0],
            ONE_STEP_CLUSTERS["cluster1"][:, 1],
            s=18,
            c="green",
            marker=".",
        )
        plt.scatter(
            0.5,
            0.05,
            s=72,
            c="green",
            marker="*",
        )
        self.font["color"] = "green"
        plt.text(
            0.55,
            0.05,
            r"D",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )

        """cluster2"""
        plt.scatter(
            ONE_STEP_CLUSTERS["cluster2"][:, 0],
            ONE_STEP_CLUSTERS["cluster2"][:, 1],
            s=18,
            c="red",
            marker=".",
        )
        plt.scatter(
            0.5,
            0.95,
            s=72,
            c="red",
            marker="*",
        )
        self.font["color"] = "red"
        plt.text(
            0.53,
            1.0,
            r"A",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        """b"""
        plt.scatter(
            0.5,
            0.49,
            s=36,
            c="black",
            marker=".",
        )
        self.font["color"] = "black"
        plt.text(
            0.53,
            0.49,
            r"b",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        """c"""
        plt.scatter(
            0.5,
            0.65,
            s=36,
            c="red",
            marker=".",
        )
        self.font["color"] = "black"
        plt.text(
            0.52,
            0.7,
            r"c",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        """画个箭头"""
        plt.arrow(
            0.5,
            0.49,
            0,
            0.16,
            length_includes_head=True,
            head_width=0.01,
            fc="b",
            ec="k",
        )
        """标题"""
        plt.title("Domino Effect", self.font, y=-0.15)

        """保存图片"""
        plt.tight_layout()
        plt.savefig("./result/diagram/domino_effect" + ".pdf", bbox_inches="tight")
        plt.show()

    def assign_base(self):
        """
        两步分配策略蓝本演示图
        Args:
        """
        # plt.axis("off")
        ax = plt.gca()
        # ax.spines["right"].set_color("none")
        # ax.spines["top"].set_color("none")

        plt.xlim(0, 14.5)
        plt.ylim(-0.2, 12.5)
        plt.xticks([])
        plt.yticks([])
        ax.set_aspect(1)

        """cluster1"""
        plt.scatter(
            TWO_STEP_CLUSTERS["cluster1"][1:, 0],
            TWO_STEP_CLUSTERS["cluster1"][1:, 1],
            s=24,
            c="green",
            marker=".",
        )
        plt.scatter(
            TWO_STEP_CLUSTERS["cluster1"][0, 0],
            TWO_STEP_CLUSTERS["cluster1"][0, 1],
            s=100,
            c="green",
            marker="^",
        )
        self.font["color"] = "green"
        plt.text(
            TWO_STEP_CLUSTERS["cluster1"][0, 0] + 0.4,
            TWO_STEP_CLUSTERS["cluster1"][0, 1] - 0.2,
            r"聚类中心",
            # fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )

        """cluster2"""
        plt.scatter(
            TWO_STEP_CLUSTERS["cluster2"][1:, 0],
            TWO_STEP_CLUSTERS["cluster2"][1:, 1],
            s=24,
            c="red",
            marker=".",
        )
        plt.scatter(
            TWO_STEP_CLUSTERS["cluster2"][0, 0],
            TWO_STEP_CLUSTERS["cluster2"][0, 1],
            s=100,
            c="red",
            marker="^",
        )
        self.font["color"] = "red"
        plt.text(
            TWO_STEP_CLUSTERS["cluster2"][0, 0] + 0.4,
            TWO_STEP_CLUSTERS["cluster2"][0, 1] - 0.2,
            r"聚类中心",
            # fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )

        """cluster3"""
        plt.scatter(
            TWO_STEP_CLUSTERS["cluster3"][1:, 0],
            TWO_STEP_CLUSTERS["cluster3"][1:, 1],
            s=24,
            c="blue",
            marker=".",
        )
        plt.scatter(
            TWO_STEP_CLUSTERS["cluster3"][0, 0],
            TWO_STEP_CLUSTERS["cluster3"][0, 1],
            s=100,
            c="blue",
            marker="^",
        )
        self.font["color"] = "blue"
        plt.text(
            TWO_STEP_CLUSTERS["cluster3"][0, 0] + 0.4,
            TWO_STEP_CLUSTERS["cluster3"][0, 1] - 0.2,
            r"聚类中心",
            # fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )

    def two_step_assign1(self):
        """
        两步样本分配策略演示图，第一张图，对应图 4-2
        """
        self.assign_base()
        plt.scatter(8, 6.5, s=24, c="blue", marker=".")

        """标注"""
        self.font["color"] = "black"
        self.font["size"] = 16
        plt.text(
            10 + 0.5,
            11 + 0.5,
            r"$\bigtriangleup$ = 聚类中心",
            # fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10 + 0.5,
            10 + 0.5,
            r"$\cdot$ = 样本点",
            # fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )

        """保存图片"""
        plt.tight_layout()
        plt.savefig(self.save_path + "two_step_assign1.pdf", bbox_inches="tight")
        plt.show()

    def two_step_assign2(self):
        """
        两步样本分配策略演示图，第二张图，对应图 4-3
        """
        self.assign_base()
        plt.scatter(8, 6.5, s=24, c="blue", marker=".")

        """突出密度中心，在密度中心画圈"""
        plt.scatter(5, 6.4, s=90, c="red", marker=".")
        self.font["color"] = "red"
        self.font["size"] = 16
        plt.text(
            5,
            6.4,
            r"$c_1$",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.scatter(8, 6.5, s=90, c="blue", marker=".")
        self.font["color"] = "blue"
        self.font["size"] = 16
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
            10 + 0.5,
            11 + 0.5,
            r"k = 5",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10 + 0.5,
            10 + 0.5,
            r"$\bigtriangleup$ = 聚类中心",
            # fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10 + 0.5,
            9 + 0.5,
            r"$\cdot$ = 样本点",
            # fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        # plt.text(
        #     10,
        #     8,
        #     r"$\star$ = ",
        #     fontdict=self.font,
        #     verticalalignment="top",
        #     horizontalalignment="left",
        # )

        """保存图片"""
        plt.tight_layout()
        plt.savefig(self.save_path + "two_step_assign2.pdf", bbox_inches="tight")
        plt.show()

    def two_step_assign3(self):
        """
        两步样本分配策略演示图，第三张图，对应图 4-4
        """
        self.assign_base()
        """cluster1"""
        plt.scatter(
            TWO_STEP_CLUSTERS["cluster1"][1:, 0],
            TWO_STEP_CLUSTERS["cluster1"][1:, 1],
            s=64,
            c="green",
            marker="+",
        )
        """cluster2"""
        plt.scatter(
            TWO_STEP_CLUSTERS["cluster2"][1:, 0],
            TWO_STEP_CLUSTERS["cluster2"][1:, 1],
            s=64,
            c="red",
            marker="+",
        )
        """cluster3"""
        plt.scatter(
            TWO_STEP_CLUSTERS["cluster3"][1:, 0],
            TWO_STEP_CLUSTERS["cluster3"][1:, 1],
            s=64,
            c="blue",
            marker="+",
        )

        """突出密度中心，在密度中心画圈"""
        plt.scatter(8, 6.5, s=64, c="blue", marker="x")
        circle4 = plt.Circle((8, 6.5), 1.3, color="b", alpha=0.7, fill=False)
        plt.gcf().gca().add_artist(circle4)
        self.font["color"] = "blue"
        self.font["size"] = 16
        plt.text(
            8,
            6.5,
            r"$c_2$",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )

        """[8, 6.5], [5.8, 6.1]"""
        plt.scatter(5.8, 6.1, s=64, c="red", marker="+")

        """标注"""
        self.font["color"] = "black"
        self.font["size"] = 16
        plt.text(
            10 + 0.5,
            11 + 0.5,
            r"k = 5",
            fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10 + 0.5,
            10 + 0.5,
            r"$\bigtriangleup$ = 聚类中心",
            # fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10 + 0.5,
            9 + 0.5,
            r"$+$ = 可信标签",
            # fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10 + 0.5,
            8 + 0.5,
            r"$\times$ = 待定标签",
            # fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        # plt.text(
        #     10,
        #     7,
        #     r"$\star$ = density peak",
        #     fontdict=self.font,
        #     verticalalignment="top",
        #     horizontalalignment="left",
        # )

        """保存图片"""
        plt.tight_layout()
        plt.savefig(self.save_path + "two_step_assign3.pdf", bbox_inches="tight")
        plt.show()

    def two_step_assign4(self):
        """
        两步样本分配策略演示图，第四张图，对应图 4-5
        """
        self.assign_base()
        """cluster1"""
        plt.scatter(
            TWO_STEP_CLUSTERS["cluster1"][1:, 0],
            TWO_STEP_CLUSTERS["cluster1"][1:, 1],
            s=64,
            c="green",
            marker="+",
        )
        """cluster2"""
        plt.scatter(
            TWO_STEP_CLUSTERS["cluster2"][1:, 0],
            TWO_STEP_CLUSTERS["cluster2"][1:, 1],
            s=64,
            c="red",
            marker="+",
        )
        """cluster3"""
        plt.scatter(
            TWO_STEP_CLUSTERS["cluster3"][1:, 0],
            TWO_STEP_CLUSTERS["cluster3"][1:, 1],
            s=64,
            c="blue",
            marker="+",
        )

        """纠错 [8, 6.5], [5.8, 6.1]"""
        plt.scatter(5.8, 6.1, s=64, c="red", marker="+")
        plt.scatter(8, 6.5, s=64, c="red", marker="+")

        """标注"""
        self.font["color"] = "black"
        self.font["size"] = 16
        plt.text(
            10 + 0.5,
            11 + 0.5,
            r"$\bigtriangleup$ = 聚类中心",
            # fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )
        plt.text(
            10 + 0.5,
            10 + 0.5,
            r"$+$ = 标签",
            # fontdict=self.font,
            verticalalignment="top",
            horizontalalignment="left",
        )

        """保存图片"""
        plt.tight_layout()
        plt.savefig(self.save_path + "two_step_assign4.pdf", bbox_inches="tight")
        plt.show()
