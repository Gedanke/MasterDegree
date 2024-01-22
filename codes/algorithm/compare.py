# -*- coding: utf-8 -*-


import os
import json
import pandas
from .setting import *
from sklearn.cluster import (
    AgglomerativeClustering,
    AffinityPropagation,
    Birch,
    DBSCAN,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
    OPTICS,
    SpectralClustering,
)


class ComBase:
    """
    使用 sklearn 类库算法进行聚类时，可以基于该类进行封装
    """

    def __init__(self, path, save_path="../../result/", num=0, params={}) -> None:
        """
        初始化相关成员
        Args:
            path (_type_): 数据集文件路径，最后一列是标签列，其余是数据列
            save_path (str, optional): 保存算法结果的路径. Defaults to "../../result/".
            num (int, optional): 类簇数量. Defaults to 0.
            params (dict, optional): 算法需要的参数. Defaults to {}.
        """
        """构造函数中的相关参数"""
        """数据文件路径"""
        self.path = path
        """保存结果的文件路径"""
        self.save_path = save_path
        """聚类类簇数，必须指定"""
        self.num = num
        """从文件路径中获取文件名(不含后缀)"""
        self.data_name = os.path.splitext(os.path.split(self.path)[-1])[0]
        """聚类算法需要的参数"""
        self.params = params

        """其他参数"""
        """算法的名称，同时也是保存最终结果的文件名称"""
        self.algorithm_name = self.data_name
        """样本集合，不含标签列"""
        self.samples = pandas.DataFrame({})
        """真实标签，默认数据的最后一列为标签"""
        self.label_true = list()
        """聚类算法预测得到的结果"""
        self.label_pred = list()
        """聚类结果，字典，里面包含各种聚类指标以及聚类的标签"""
        self.cluster_result = {"label": list()}
        """获取数据集部分信息"""
        self.init_points_msg()

        """聚类函数"""
        self.cluster()

    def init_points_msg(self):
        """
        加载数据，输入的数据一定要满足预设要求
        """
        """读取 csv 文件全部内容"""
        self.samples = pandas.read_csv(self.path)
        col = list(self.samples.columns)
        """self.samples，最后一列为标签列"""
        self.label_true = self.samples[col[-1]].tolist()
        """self.samples 其余列为数据列，不包括标签列"""
        self.samples = numpy.array(self.samples.loc[:, col[0:-1]])

    def cluster(self):
        """
        聚类算法的实现，使用 sklearn 中的方法进行聚类，将聚类结果保存
        """

    def get_file_path(self, name):
        """
        创建文件夹，返回保存文件路径
        Args:
            name (_type_): 算法名称
        Returns:
            _type_: 文件路径
        """
        """创建以该数据集命名的文件夹"""
        path = self.save_path + "result/" + self.data_name + "/"
        if not os.path.isdir(path):
            """创建文件夹"""
            os.mkdir(path)

        """创建以该算法名命名的文件夹"""
        path += name + "/"
        if not os.path.isdir(path):
            """创建文件夹"""
            os.mkdir(path)

        """文件名"""
        path += self.algorithm_name + ".json"

        return path

    def save_result(self, name):
        """
        将聚类结果与聚类指标保存到文件中
        Args:
            name (_type_): 算法名称
        """
        """保存聚类结果"""
        self.cluster_result["label"] = self.label_pred.tolist()
        """聚类指标写入到 json 文件中"""
        self.cluster_result.update(
            cluster_index(self.samples, self.label_true, self.label_pred, self.num != 0)
        )

        """写入文件"""
        with open(self.get_file_path(name), "w", encoding="utf-8") as f:
            f.write(json.dumps(self.cluster_result, ensure_ascii=False))


class ComAC(ComBase):
    """
    层次聚类
    AgglomerativeClustering 算法的封装
    重要参数 affinity, linkage
    Args:
        ComBase (_type_): 基类
    """

    def __init__(self, path, save_path="../../result/", num=0, params={}) -> None:
        """
        初始化相关成员
        Args:
            path (_type_): 数据集文件路径
            save_path (str, optional): 保存算法结果的路径. Defaults to "../../result/".
            num (int, optional): 类簇数量. Defaults to 0.
            params (dict, optional): 算法需要的参数. Defaults to {}.
        """
        super(ComAC, self).__init__(path, save_path, num, params)

    def cluster(self):
        """
        AgglomerativeClustering 聚类算法
        """
        """确定文件名"""
        self.algorithm_name = "ac"
        """AgglomerativeClustering 聚类算法"""
        if "affinity" in self.params.keys() and "linkage" in self.params.keys():
            algorithm = AgglomerativeClustering(
                n_clusters=self.num,
                affinity=self.params["affinity"],
                linkage=self.params["linkage"],
            )
            self.algorithm_name += (
                "__af_"
                + str(self.params["affinity"])
                + "__la_"
                + str(self.params["linkage"])
            )
        elif "linkage" not in self.params.keys() and "affinity" in self.params.keys():
            algorithm = AgglomerativeClustering(
                n_clusters=self.num, affinity=self.params["affinity"]
            )
            self.algorithm_name += "__af_" + str(self.params["affinity"])
        elif "linkage" in self.params.keys() and "affinity" not in self.params.keys():
            algorithm = AgglomerativeClustering(
                n_clusters=self.num, linkage=self.params["linkage"]
            )
            self.algorithm_name += "__la_" + str(self.params["linkage"])
        else:
            algorithm = AgglomerativeClustering(n_clusters=self.num)

        """保留 affinity 与 linkage"""
        self.cluster_result["affinity"] = algorithm.affinity
        self.cluster_result["linkage"] = algorithm.linkage
        """预测标签"""
        self.label_pred = algorithm.fit_predict(self.samples)
        """保存聚类结果"""
        self.save_result("agglomerativeClustering")


class ComAP(ComBase):
    """
    AffinityPropagation 算法的封装
    Args:
        ComBase (_type_): 基类
    """

    def __init__(self, path, save_path="../../result/", num=0, params={}) -> None:
        """
        初始化相关成员
        Args:
            path (_type_): 数据集文件路径
            save_path (str, optional): 保存算法结果的路径. Defaults to "../../result/".
            num (int, optional): 类簇数量. Defaults to 0.
            params (dict, optional): 算法需要的参数. Defaults to {}.
        """
        super(ComAP, self).__init__(path, save_path, num, params)

    def cluster(self):
        """
        AffinityPropagation 聚类算法
        """
        """确定文件名"""
        self.algorithm_name = "ap"
        """AP 聚类"""
        if "damping" in self.params.keys():
            algorithm = AffinityPropagation(preference=self.params["damping"])
            self.algorithm_name += "__damping_" + str(self.params["damping"])
        else:
            algorithm = AffinityPropagation()

        """保留 damping"""
        self.cluster_result["damping"] = algorithm.damping
        """预测标签"""
        self.label_pred = algorithm.fit_predict(self.samples)
        """加入聚类中心坐标索引"""
        self.cluster_result["center"] = [
            int(numpy.where(center == self.samples)[0][0])
            for center in algorithm.cluster_centers_
        ]

        """预测的聚类结果超过一类，才有价值"""
        if len((set(self.label_pred))) > 1:
            """保存聚类结果"""
            self.save_result("affinityPropagation")


class ComBirch(ComBase):
    """
    Birch 算法的封装
    Args:
        ComBase (_type_): 基类
    """

    def __init__(self, path, save_path="../../result/", num=0, params={}) -> None:
        """
        初始化相关成员
        Args:
            path (_type_): 数据集文件路径
            save_path (str, optional): 保存算法结果的路径. Defaults to "../../result/".
            num (int, optional): 类簇数量. Defaults to 0.
            params (dict, optional): 算法需要的参数. Defaults to {}.
        """
        super(ComBirch, self).__init__(path, save_path, num, params)

    def cluster(self):
        """
        Birch 聚类算法
        """
        """确定文件名"""
        self.algorithm_name = "bi"
        """Birch 聚类算法"""
        if "threshold" in self.params.keys():
            algorithm = Birch(n_clusters=self.num, threshold=self.params["threshold"])
            self.algorithm_name += "__ts_" + str(self.params["threshold"])
        else:
            algorithm = Birch(n_clusters=self.num)

        """保留 threshold"""
        self.cluster_result["threshold"] = algorithm.threshold
        """预测标签"""
        self.label_pred = algorithm.fit_predict(self.samples)
        """保存聚类结果"""
        self.save_result("birch")


class ComDBSCAN(ComBase):
    """
    DBSCAN 算法的封装
    重要参数：
    eps，min_samples
    Args:
        ComBase (_type_): 基类
    """

    def __init__(self, path, save_path="../../result/", num=0, params={}) -> None:
        """
        初始化相关成员
        Args:
            path (_type_): 数据集文件路径
            save_path (str, optional): 保存算法结果的路径. Defaults to "../../result/".
            num (int, optional): 类簇数量. Defaults to 0.
            params (dict, optional): 算法需要的参数. Defaults to {}.
        """
        super(ComDBSCAN, self).__init__(path, save_path, num, params)

    def cluster(self):
        """
        DBSCAN 聚类算法
        """
        """确定文件名"""
        self.algorithm_name = "db"
        """DBSCAN 聚类算法"""
        if "eps" in self.params.keys() and "min_samples" in self.params.keys():
            algorithm = DBSCAN(
                eps=self.params["eps"], min_samples=self.params["min_samples"]
            )
            self.algorithm_name += (
                "__eps_"
                + str(self.params["eps"])
                + "__ms_"
                + str(self.params["min_samples"])
            )
        else:
            algorithm = DBSCAN()

        """保留 eps 与 min_samples"""
        self.cluster_result["eps"] = algorithm.eps
        self.cluster_result["min_samples"] = algorithm.min_samples
        """预测标签"""
        self.label_pred = algorithm.fit_predict(self.samples)
        """预测的聚类结果超过一类，才有价值"""
        if len((set(self.label_pred))) > 0:
            """保存聚类结果"""
            self.save_result("dbscan")


class ComKMeans(ComBase):
    """
    KNeans 算法的封装
    Args:
        ComBase (_type_): 基类
    """

    def __init__(self, path, save_path="../../result/", num=0, params={}) -> None:
        """
        初始化相关成员
        Args:
            path (_type_): 数据集文件路径
            save_path (str, optional): 保存算法结果的路径. Defaults to "../../result/".
            num (int, optional): 类簇数量. Defaults to 0.
            params (dict, optional): 算法需要的参数. Defaults to {}.
        """
        super(ComKMeans, self).__init__(path, save_path, num, params)

    def cluster(self):
        """
        KMeans 聚类算法
        不管是 KMeans 类还是 MiniBatchKMeans 类，我们只提供 n_clusters 这个参数
        其他所有参数均为默认，特别地，init 是 k-means++
        """
        """判断是否进行大数据优化"""
        if "MiniBatch" in self.params.keys() and self.params["MiniBatch"]:
            """使用 MiniBatchKMeans"""
            algorithm = MiniBatchKMeans(n_clusters=self.num)
            """确定文件名"""
            self.algorithm_name = "miniBatchKMeans"
        else:
            """使用 KMeans"""
            algorithm = KMeans(n_clusters=self.num)
            """确定文件名"""
            self.algorithm_name = "kmeans"

        """预测标签"""
        self.label_pred = algorithm.fit_predict(self.samples)
        """加入聚类中心坐标"""
        self.cluster_result["center"] = numpy.around(
            algorithm.cluster_centers_, decimals=3
        ).tolist()
        """保存聚类结果"""
        self.save_result("kmeans")


class ComMeanShit(ComBase):
    """
    MeanShift 算法的封装
    Args:
        ComBase (_type_): 基类
    """

    def __init__(self, path, save_path="../../result/", num=0, params={}) -> None:
        """
        初始化相关成员
        Args:
            path (_type_): 数据集文件路径
            save_path (str, optional): 保存算法结果的路径. Defaults to "../../result/".
            num (int, optional): 类簇数量. Defaults to 0.
            params (dict, optional): 算法需要的参数. Defaults to {}.
        """
        super(ComMeanShit, self).__init__(path, save_path, num, params)

    def cluster(self):
        """
        MeanShift 聚类算法
        """
        """确定文件名"""
        self.algorithm_name = "ms"
        """bandwidth 设置"""
        if "bandwidth" in self.params.keys():
            algorithm = MeanShift(bandwidth=self.params["bandwidth"])
            self.algorithm_name += "__bd_" + str(self.params["bandwidth"])
        else:
            algorithm = MeanShift()

        """保留 bandwidth"""
        self.cluster_result["bandwidth"] = algorithm.bandwidth
        """预测标签"""
        self.label_pred = algorithm.fit_predict(self.samples)
        """加入聚类中心坐标"""
        self.cluster_result["center"] = algorithm.cluster_centers_.tolist()
        """预测的聚类结果超过一类，才有价值"""
        if len((set(self.label_pred))) > 1:
            """保存聚类结果"""
            self.save_result("meanShift")


class ComOPTICS(ComBase):
    """
    OPTICS 算法的封装
    重要参数：
    eps，min_samples
    Args:
        ComBase (_type_): 基类
    """

    def __init__(self, path, save_path="../../result/", num=0, params={}) -> None:
        """
        初始化相关成员
        Args:
            path (_type_): 数据集文件路径
            save_path (str, optional): 保存算法结果的路径. Defaults to "../../result/".
            num (int, optional): 类簇数量. Defaults to 0.
            params (dict, optional): 算法需要的参数. Defaults to {}.
        """
        super(ComOPTICS, self).__init__(path, save_path, num, params)

    def cluster(self):
        """
        OPTICS 聚类算法
        """
        """确定文件名"""
        self.algorithm_name = "op"
        """OPTICS 聚类算法"""
        if "eps" in self.params.keys() and "min_samples" in self.params.keys():
            algorithm = OPTICS(
                eps=self.params["eps"], min_samples=self.params["min_samples"]
            )
            self.algorithm_name += (
                "__eps_"
                + str(self.params["eps"])
                + "__ms_"
                + str(self.params["min_samples"])
            )
        else:
            algorithm = OPTICS()

        """保留 eps 与 min_samples"""
        self.cluster_result["eps"] = algorithm.eps
        self.cluster_result["min_samples"] = algorithm.min_samples
        """预测标签"""
        self.label_pred = algorithm.fit_predict(self.samples)
        """预测的聚类结果超过一类，才有价值"""
        if len((set(self.label_pred))) > 1:
            """保存聚类结果"""
            self.save_result("optics")


class ComSC(ComBase):
    """
    SpectralClustering 算法的封装
    affinity 默认为 rbf，其参数通过 gamma 调节
    Args:
        ComBase (_type_): 基类
    """

    def __init__(self, path, save_path="../../result/", num=0, params={}) -> None:
        """
        初始化相关成员
        Args:
            path (_type_): 数据集文件路径
            save_path (str, optional): 保存算法结果的路径. Defaults to "../../result/".
            num (int, optional): 类簇数量. Defaults to 0.
            params (dict, optional): 算法需要的参数. Defaults to {}.
        """
        super(ComSC, self).__init__(path, save_path, num, params)

    def cluster(self):
        """
        SpectralClustering 聚类算法
        """
        """确定文件名"""
        self.algorithm_name = "sc"
        """SpectralClustering 聚类算法"""
        if "gamma" in self.params.keys():
            algorithm = SpectralClustering(
                n_clusters=self.num, gamma=self.params["gamma"]
            )
            """确定文件名"""
            self.algorithm_name += "__gamma_" + str(self.params["gamma"])
        else:
            algorithm = SpectralClustering(n_clusters=self.num)

        """保留 gamma"""
        self.cluster_result["gamma"] = algorithm.gamma
        """预测标签"""
        self.label_pred = algorithm.fit_predict(self.samples)
        """保存聚类结果"""
        self.save_result("spectralClustering")
