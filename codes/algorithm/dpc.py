# -*- coding: utf-8 -*-


import os
import math
import json
import pandas
import scipy.cluster.hierarchy as sch
from .setting import *


class Dpc:
    """
    经典 DPC 算法基类
    """

    def __init__(
        self,
        path,
        save_path="./result/",
        num=0,
        dc_percent=1,
        dc_method=1,
        rho_method=0,
        delta_method=0,
        distance_method="euclidean",
        center=[],
        use_halo=False,
        params={},
    ) -> None:
        """
        初始化相关成员，方法索引均从 0 开始
        Args:
            path (_type_): 文件完整路径，除图片类型文件外，均为 csv 文件，数据均满足，最后一列为标签列，其余为数据列.
            save_path (str, optional): 保存结果的路径. Defaults to "./result/".
            num (int, optional): 聚类类簇数. Defaults to 0.
            dc_percent (int, optional): 截断距离百分比数. Defaults to 1.
            dc_method (int, optional): 截断距离计算方法. Defaults to 1.
            rho_method (int, optional): 局部密度计算方法. Defaults to 0.
            delta_method (int, optional): 相对距离计算方法. Defaults to 0.
            distance_method (str, optional): 度量方式. Defaults to 'euclidean'.
            center (list, optional): 聚类中心，可以人为指定，但为了统一起见，不建议这样做. Defaults to [].
            use_halo (bool, optional): 是否计算光晕点. Defaults to False.
            params (dict, optional): 改进算法需要的其他参数. Defaults to {}.
        """
        """构造函数中的相关参数"""
        """文件完整路径"""
        self.path = path
        """保存结果文件的路径"""
        self.save_path = save_path
        """从文件路径中获取文件名(不含后缀)"""
        self.data_name = os.path.splitext(os.path.split(self.path)[-1])[0]
        """聚类类簇数，最好指定，也可以从文件中读取得到(默认从最后一列中读取)"""
        self.num = int(num)
        """截断距离百分比数"""
        self.dc_percent = float(dc_percent)
        """截断距离计算方法"""
        self.dc_method = dc_method
        """局部密度计算方法"""
        self.rho_method = rho_method
        """相对距离计算方法"""
        self.delta_method = delta_method
        """距离度量方式"""
        self.distance_method = distance_method
        """聚类中心"""
        self.center = numpy.array(center)
        """是否计算光晕点"""
        self.use_halo = use_halo
        """算法需要的其他参数"""
        self.params = params

        """其他参数"""
        """边界域中密度最大的点"""
        self.border_b = list()
        """数据集的所有样本点，不包括标签列"""
        self.samples = pandas.DataFrame({})
        """真实标签列"""
        self.label_true = list()
        """样本个数"""
        self.samples_num = 0
        """属性个数，不包含标签列"""
        self.features_num = 0
        """聚类结果"""
        self.label_pred = list()
        """距离矩阵(样本间的度量方式以后续需要的为主)"""
        self.dis_matrix = pandas.DataFrame({})
        """聚类结果指标"""
        self.cluster_result = dict()
        """将 dc，rho，delta 设置为成员，方便作图"""
        self.dc = 0
        self.rho = None
        self.delta = None

        """保存文件相关参数"""
        """算法名称"""
        self.algorithm_name = "dpc"
        """由于已经创建了以该文件名命名的文件夹，对于文件名只需要添加相关参数"""
        self.file_name = (
            "dcp_"
            + str(self.dc_percent)
            + "__dcm_"
            + str(self.dc_method)
            + "__rhom_"
            + str(self.rho_method)
            + "__dem_"
            + str(self.delta_method)
        )
        """use_halo 几乎不用，只有用的时候才加上"""
        if self.use_halo:
            self.file_name += "__ush_" + str(int(self.use_halo))

        """聚类函数"""
        self.cluster()

    def cluster(self):
        """
        聚类过程
        之后再将传入成员变量作为函数参数的方法进行修改，edit
        """
        """获取数据集相对固定的成员信息：样本点，样本数"""
        self.init_points_msg()
        """获取数据集其他相关的成员信息：距离矩阵，距离列表，最大距离，最小距离"""
        dis_array, max_dis, min_dis = self.load_points_msg()
        """计算截断距离"""
        self.dc = self.get_dc(dis_array, max_dis, min_dis)
        """计算局部密度"""
        self.rho = self.get_rho(self.dc)
        """计算相对距离 delta"""
        self.delta = self.get_delta(self.rho)
        """确定聚类中心，计算 gamma(局部密度于相对距离的乘积)"""
        gamma = self.get_center(self.rho, self.delta)
        """非聚类中心样本点分配"""
        cluster_results = self.assign_samples(self.rho, self.center)
        """光晕点"""
        if self.use_halo:
            """默认不使用"""
            cluster_results, halo = self.get_halo(self.rho, cluster_results, self.dc)
            self.cluster_result["halo"] = halo

        """获取聚类结果"""
        self.gain_label_pred(cluster_results)
        """聚类结果，指标写入到 json 文件中"""
        self.save_result()

    def init_points_msg(self):
        """
        辅助 self.load_points_msg 完成部分固定信息的初始化
        """
        """读取 csv 文件"""
        self.samples = pandas.read_csv(self.path)
        """标签列表"""
        cols = list(self.samples.columns)
        """特征个数"""
        self.features_num = len(cols) - 1
        """样本个数"""
        self.samples_num = len(self.samples)
        """真实标签列"""
        self.label_true = self.samples[cols[-1]].tolist()
        """数据列，不包含标签列"""
        self.samples = self.samples.loc[:, cols[0:-1]]
        """初始化距离矩阵"""
        self.dis_matrix = pandas.DataFrame(
            numpy.zeros((self.samples_num, self.samples_num))
        )

    def load_points_msg(self):
        """
        获取数据集相关信息，如距离矩阵，欧式距离列表，最大距离，最小距离
        Args:
        Returns:
            dis_array (_type_): 样本间距离的矩阵
            max_dis (_type_): 样本间最大距离
            min_dis (_type_): 样本间最小距离
        """
        """默认为欧式距离，可以根据不同的度量方法调用不同的方法生成距离矩阵，最大距离，最小距离等信息，这里只给出经典 DPC 算法的实现"""
        return self.distance_standard()

    def distance_standard(self):
        """
        根据度量方法生成距离矩阵等信息
        Args:
        Returns:
            dis_array (_type_): 样本间距离的矩阵
            max_dis (_type_): 样本间最大距离
            min_dis (_type_): 样本间最小距离
        """
        """维度为 self.samples_num * (self.samples_num - 1) / 2，默认为 euclidean"""
        dis_array = sch.distance.pdist(self.samples, self.distance_method)

        """处理距离矩阵"""
        num = 0
        for i in range(self.samples_num):
            for j in range(i + 1, self.samples_num):
                """赋值"""
                self.dis_matrix.at[i, j] = dis_array[num]
                """处理对角元素"""
                self.dis_matrix.at[j, i] = self.dis_matrix.at[i, j]
                num += 1

        """最大距离"""
        max_dis = self.dis_matrix.max().max()
        """最小距离"""
        min_dis = self.dis_matrix.min().min()

        return dis_array, max_dis, min_dis

    def get_dc(self, dis_array, max_dis, min_dis):
        """
        根据不同的 self.dc_method 计算截断距离
        Args:
            dis_array (_type_): 样本间距离的矩阵
            max_dis (_type_): 样本间最大距离
            min_dis (_type_): 样本间最小距离
        Returns:
            dc (_type_): 截断距离
        """
        """最低最高百分比数"""
        lower = self.dc_percent / 100
        upper = (self.dc_percent + 1) / 100
        dc = 0.0

        """判断计算截断距离的方法，默认是该方法"""
        if self.dc_method == 0:
            while 1:
                dc = (min_dis + max_dis) / 2
                """上三角矩阵"""
                neighbors_percent = len(dis_array[dis_array < dc]) / (
                    ((self.samples_num - 1) ** 2) / 2
                )

                if lower <= neighbors_percent <= upper:
                    return dc
                elif neighbors_percent > upper:
                    max_dis = dc
                elif neighbors_percent < lower:
                    min_dis = dc
        elif self.dc_method == 1:
            """实验中默认该方法"""
            dis_array_ = dis_array.copy()
            dis_array_.sort()
            """取第 self.dc_percent 个距离作为截断距离"""
            dc = dis_array_[
                int(
                    float(self.dc_percent / 100.0)
                    * self.samples_num
                    * (self.samples_num - 1)
                    / 2
                )
            ]
        elif self.dc_method == 2:
            """如果对截断距离计算有所改进，可以重写该方法"""
            pass

        return dc

    def get_rho(self, dc):
        """
        计算局部密度
        Args:
            dc (_type_): 截断距离
        Returns:
            rho (_type_): 每个样本点的局部密度
        """
        """每个样本点的局部密度"""
        rho = numpy.zeros(self.samples_num)

        if self.rho_method == 0:
            """高斯核，实验中默认该方法"""
            for i in range(self.samples_num):
                for j in range(self.samples_num):
                    if i != j:
                        rho[i] += math.exp(-((self.dis_matrix.at[i, j] / dc) ** 2))
        elif self.rho_method == 1:
            """到样本点 i 距离小于 dc 的点数量"""
            for i in range(self.samples_num):
                rho[i] = (
                    len(self.dis_matrix.loc[i, :][self.dis_matrix.loc[i, :] < dc]) - 1
                )
        elif self.rho_method == 2:
            """排除异常值"""
            for i in range(self.samples_num):
                n = int(self.samples_num * 0.05)
                """选择前 n 个离 i 最近的样本点"""
                rho[i] = math.exp(
                    -(self.dis_matrix.loc[i].sort_values().values[:n].sum() / (n - 1))
                )
        elif self.rho_method == 3:
            """如果对局部密度计算有所改进，可以重写该方法"""
            pass

        return rho

    def get_delta(self, rho):
        """
        计算相对距离
        Args:
            rho (_type_): 局部密度
        Returns:
            delta (_type_): 每个样本点的相对距离
        """
        """每个样本点的相对距离"""
        delta = numpy.zeros(self.samples_num)

        """考虑局部密度 rho 是否存在多个最大值"""
        if self.delta_method == 0:
            """考虑 rho 相同且同时为最大的情况，实验中默认该方法"""
            """将局部密度从大到小排序，返回对应的索引值"""
            rho_order_idx = rho.argsort()[-1::-1]
            """即 rho_order_idx 存放的是第 i 大元素 rho 的索引，而不是对应的值"""
            for i in range(1, self.samples_num):
                """第 i 个样本"""
                rho_idx = rho_order_idx[i]
                """j < i 的排序索引值或者说 rho > i 的索引列表"""
                j_list = rho_order_idx[:i]
                """返回第 idx_rho 行，j_list 列中最小值对应的索引"""
                min_dis_idx = self.dis_matrix.loc[rho_idx, j_list].idxmin()
                """i 的相对距离，即比 i 大，且离 i 最近的距离"""
                delta[rho_idx] = self.dis_matrix.at[rho_idx, min_dis_idx]
            """相对距离最大的点"""
            delta[rho_order_idx[0]] = delta.max()
        elif self.delta_method == 1:
            """不考虑 rho 相同且同时为最大的情况"""
            for i in range(self.samples_num):
                """第 i 个样本的局部密度"""
                rho_i = rho[i]
                """局部密度比 rho_i 大的点集合的索引"""
                j_list = numpy.where(rho > rho_i)[0]
                if len(j_list) == 0:
                    """rho_i 是 rho 中局部密度最大的点"""
                    delta[i] = self.dis_matrix.loc[i, :].max()
                else:
                    """非局部密度最大的点，寻找局部密度大于 i 且离 i 最近的点的索引"""
                    min_dis_idx = self.dis_matrix.loc[i, j_list].idxmin()
                    delta[i] = self.dis_matrix.at[i, min_dis_idx]
        elif self.delta_method == 2:
            """如果对相对距离计算有所改进，可以直接重写该方法"""
            pass

        return delta

    def get_center(self, rho, delta):
        """
        获取聚类中心，计算 gamma
        Args:
            rho (_type_): 局部密度
            delta (_type_): 相对距离
        Returns:
            gamma (_type_): rho * delta
        """
        gamma = rho * delta

        """对 gamma 排序"""
        gamma = pandas.DataFrame(gamma, columns=["gamma"]).sort_values(
            "gamma", ascending=False
        )

        """没有指定聚类中心，self.center 为空"""
        if len(self.center) == 0:
            if self.num > 0:
                """取 gamma 中前 self.samples_num 个点作为聚类中心，默认使用该方法"""
                self.center = numpy.array(gamma.index)[: self.num]
            else:
                """采用其他方法"""
                # center = gamma[gamma.gamma > threshold].loc[:, "gamma"].index

        """保存聚类中心"""
        self.cluster_result["center"] = self.center.tolist()

        return gamma

    def assign_samples(self, rho, center):
        """
        非聚类中心样本点分配策略
        Args:
            rho (_type_): 局部密度
            center (_type_): 聚类中心样本点，有时候可以通过指定特定的点作为聚类中心，为了统一起见，不推荐这样做，但还是保留了这个选择
        Returns:
            (dict(center: str, points: list())): 聚类结果
        """
        """默认为一步分配策略，可以根据不同的分配方法分配非距离中心样本点，这里只给出经典 DPC 算法的实现"""
        return self.assign(rho, center)

    def assign(self, rho, center):
        """
        DPC 算法的非聚类中心样本点分配
        Args:
            rho (_type_): 局部密度
            center (_type_): 聚类中心样本点
        Returns:
            cluster_results (dict(center: str, points: list())): 聚类结果
        """
        """链式分配方法(顺藤摸瓜)"""
        cluster_results = dict()
        """键为聚类中心索引，值为归属聚类中心的所有样本点，包括聚类中心本身"""
        for c in center:
            cluster_results[c] = list()

        """link 的键为当前样本，值为离当前点最近的样本点"""
        link = dict()
        """局部密度从大到小排序，返回索引值"""
        order_rho_idx = rho.argsort()[-1::-1]
        for i, v in enumerate(order_rho_idx):
            if v in center:
                """聚类中心"""
                link[v] = v
                continue
            """非聚类中心的点"""
            """前 i 个局部密度点的排序索引值，也就是局部密度大于 rho[v] 的索引列表"""
            rho_idx = order_rho_idx[:i]
            """在局部密度大于 rho[v] 的点中，距离从小到大排序的第一个索引，也是离得最近的点(不一定是聚类中心)"""
            link[v] = self.dis_matrix.loc[v, rho_idx].sort_values().index.tolist()[0]

        """分配所有样本点"""
        for k, v in link.items():
            """使用 c 纪录离 k 最近的点 v"""
            c = v
            """c 不是聚类中心"""
            while c not in center:
                """c 更新为离 c 最近的点 link[c]，一步步迭代，顺藤摸瓜，直到找到 c 对应的聚类中心"""
                c = link[c]
            """c 是聚类中心，分配当前 k 到 c 中"""
            cluster_results[c].append(k)

        """最近中心分配方法"""
        """
        for i in range(self.samples_num):
            c = self.dis_matrix.loc[i, center].idxmin()
            cluster_results[c].append(i)
        """

        return cluster_results

    def get_halo(self, rho, cluster_result, dc):
        """
        获取光晕点
        Args:
            rho (_type_): 局部密度
            cluster_result (_type_): 聚类结果
            dc (_type_): 截断距离
        Returns:
            cluster_result (dict(center: str, points: list())): 聚类结果
            halo (list): 光晕点
        """
        """所有样本点"""
        all_points = set(list(range(self.samples_num)))

        for c, points in cluster_result.items():
            """属于其他聚类的点"""
            others_points = list(set(all_points) - set(points))
            border = list()

            for point in points:
                """到其他聚类中心点的距离小于 dc"""
                if self.dis_matrix.loc[point, others_points].min() < dc:
                    border.append(point)
            if len(border) == 0:
                continue

            """边界域中密度最大的值"""
            # rbo_b = rho[border].max()
            """边界域中密度最大的点"""
            point_b = border[rho[border].argmax()]
            self.border_b.append(point_b)
            """边界域最大密度"""
            rho_b = rho[point_b]
            """筛选可靠性高的点"""
            filter_points = numpy.where(rho >= rho_b)[0]
            """该聚类中可靠性高的点"""
            points = list(set(filter_points) & set(points))
            cluster_result[c] = points

        """halo"""
        cluster_points = set()
        for c, points in cluster_result.items():
            cluster_points = cluster_points | set(points)
        """光晕点"""
        halo = list(set(all_points) - cluster_points)

        return cluster_result, halo

    def gain_label_pred(self, cluster_result):
        """
        从 cluster_result 中获聚类标签
        Args:
            cluster_result (_type_): 聚类结果
        """
        """预分配空间"""
        self.label_pred = [-1 for _ in range(self.samples_num)]
        """以下两种方法均可以，数据集是必须含有标签的，可以从标签列获取，也可以从 0 到 self.num - 1 建立索引"""
        """存在标签，则为 self.label_true 真实标签集合列表"""
        label_true_list = list(set(self.label_true))
        # """不存在标签，以 self.num 建立从 0 到 self.num - 1 的索引作为标签"""
        # label_true_list = list(range(0, self.num))

        idx = 0
        """得到聚类标签"""
        for c, points in cluster_result.items():
            for point in points:
                self.label_pred[point] = label_true_list[idx]
            idx += 1

    def get_file_path(self):
        """
        保存文件路径
        Returns:
            path (_type_): 文件路径
        """
        """创建以该数据集命名的文件夹"""
        path = self.save_path + "result/" + self.data_name + "/"
        """判断文件夹是否存在"""
        if not os.path.isdir(path):
            """创建文件夹"""
            os.mkdir(path)

        """创建以该算法名命名的文件夹"""
        path += self.algorithm_name + "/"
        """判断文件夹是否存在"""
        if not os.path.isdir(path):
            """创建文件夹"""
            os.mkdir(path)

        """文件名"""
        path += self.file_name + ".json"

        return path

    def save_result(self):
        """
        得出聚类结果，以字典形式储存
        """
        """只保存标签列"""
        self.cluster_result["label"] = self.label_pred
        """合并聚类指标"""
        self.cluster_result.update(
            cluster_index(self.samples, self.label_true, self.label_pred, self.num != 0)
        )
        """写入文件"""
        with open(self.get_file_path(), "w", encoding="utf-8") as f:
            f.write(json.dumps(self.cluster_result, ensure_ascii=False))
