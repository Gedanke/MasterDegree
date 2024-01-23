# -*- coding: utf-8 -*-


from .dpcs import *
from collections import Counter


"""
论文中涉及到三个 DPC 改进算法
以及将三个改进变为可选选项与经典 DPC 算法融合
"""


class DpcCkrod(Dpc):
    """
    改进了距离度量的 DPC 算法
    Args:
        Dpc (_type_): DPC 算法基类，也可以从 DpcD 继承
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
        distance_method="ckrod",
        center=[],
        use_halo=False,
        params={},
    ) -> None:
        """
        初始化相关成员
        Args:
            path (_type_): 文件完整路径，除图片类型文件外，均为 csv 文件，数据均满足，最后一列为标签列，其余为数据列.
            save_path (str, optional): 保存结果的路径. Defaults to "./result/".
            num (int, optional): 聚类类簇数. Defaults to 0.
            dc_percent (int, optional): 截断距离百分比数. Defaults to 1.
            dc_method (int, optional): 截断距离计算方法. Defaults to 1.
            rho_method (int, optional): 局部密度计算方法. Defaults to 0.
            delta_method (int, optional): 相对距离计算方法. Defaults to 0.
            distance_method (str, optional): 度量方式. Defaults to 'ckrod'.
            center (list, optional): 聚类中心，可以人为指定，但为了统一起见，不建议这样做. Defaults to [].
            use_halo (bool, optional): 是否计算光晕点. Defaults to False.
            params (dict, optional): 改进算法需要的其他参数. Defaults to {}.
        """
        super(DpcCkrod, self).__init__(
            path,
            save_path,
            num,
            dc_percent,
            dc_method,
            rho_method,
            delta_method,
            distance_method,
            center,
            use_halo,
            params,
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
        """算法名称"""
        self.algorithm_name = "dpc_ckrod"
        """由于已经创建了以该文件名命名的文件夹，对于文件名只需要添加相关参数"""
        """度量方法默认为 ckrod"""
        self.file_name = (
            "dcp_"
            + str(self.dc_percent)
            + "__dcm_"
            + str(self.dc_method)
            + "__rhom_"
            + str(self.rho_method)
            + "__dem_"
            + str(self.delta_method)
            + "__mu_"
            + str(self.params["mu"])
        )
        """use_halo 几乎不用，只有用的时候才加上"""
        if self.use_halo:
            self.file_name += "__ush_" + str(int(self.use_halo))

        return self.distance_ckrod()

    def distance_ckrod(self):
        """
        ROD 系列度量方式
        Returns:
            dis_array (_type_): 样本间距离的矩阵
            max_dis (_type_): 样本间最大距离
            min_dis (_type_): 样本间最小距离
        """
        """先根据欧式距离生成所有样本的距离列表"""
        dis_array = sch.distance.pdist(self.samples, "euclidean")
        """这里采用的方法是先深拷贝一份 self.dis_matrix"""
        euclidean_table = self.dis_matrix.copy()

        """处理距离矩阵"""
        num = 0
        for i in range(self.samples_num):
            for j in range(i + 1, self.samples_num):
                """赋值"""
                self.dis_matrix.at[i, j] = dis_array[num]
                """处理对角元素"""
                self.dis_matrix.at[j, i] = self.dis_matrix.at[i, j]
                num += 1

        """对 euclidean_table 使用 argsort()，该函数会对矩阵的每一行从小到大排序，返回的是 euclidean_table 中索引"""
        rank_order_table = numpy.array(euclidean_table).argsort()

        """优化下度量计算"""
        """用新的度量方式得到的样本间距离覆盖掉 dis_array"""
        dis_array = [
            self.ckrod_fun(i, j, rank_order_table, euclidean_table)
            for i in range(self.samples_num)
            for j in range(i + 1, self.samples_num)
        ]

        """对距离矩阵进行处理"""
        num = 0
        for i in range(self.samples_num):
            for j in range(i + 1, self.samples_num):
                """self.dis_matrix 内存放样本间的距离"""
                self.dis_matrix.at[i, j] = dis_array[num]
                """处理对角元素"""
                self.dis_matrix.at[j, i] = self.dis_matrix.at[i, j]
                num += 1

        """最大距离"""
        max_dis = self.dis_matrix.max().max()
        """最小距离"""
        min_dis = self.dis_matrix.min().min()

        return dis_array, max_dis, min_dis

    def ckrod_fun(self, i, j, rank_order_table, euclidean_table):
        """
        rod 及其改进算法
        Args:
            i (_type_): 第 i 个样本
            j (_type_): 第 j 个样本
            rank_order_table (_type_): 排序距离表
            euclidean_table (_type_): 欧式距离表
        Returns:
            dis (_type_): i 与 j 之间的距离
        """
        """xi 与 xj 之间的距离"""
        dis = -1
        """第 i 个样本"""
        x1 = rank_order_table[i, :]
        """第 j 个样本"""
        x2 = rank_order_table[j, :]

        """rod 及其改进算法需要的一些前置条件"""
        """x1 样本的 id"""
        id_x1 = x1[0]
        """x2 样本的 id"""
        id_x2 = x2[0]
        """o_a_b，b 在 a 的排序列表中的索引位置"""
        o_a_b = numpy.where(x1 == id_x2)[0][0]
        """o_b_a，a 在 b 的排序列表中的索引位置"""
        o_b_a = numpy.where(x2 == id_x1)[0][0]

        """改进 krod"""
        l_a_b = (o_a_b + o_b_a) / (self.samples_num - 1) / self.features_num
        """高斯核"""
        k_a_b = math.exp(-((euclidean_table.at[i, j] / self.params["mu"]) ** 2))
        """ckrod，改进 krod 与高斯核相结合"""
        dis = l_a_b / k_a_b

        return dis


class DpcIRho(DpcCkrod):
    """
    结合了新的距离度量方式，改进了局部密度距离的 DPC 算法
    Args:
        DpcCkrod (_type_): 改进了距离度量的 DPC 算法
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
        distance_method="ckrod",
        center=[],
        use_halo=False,
        params={},
    ) -> None:
        """
        初始化相关成员
        Args:
            path (_type_): 文件完整路径，除图片类型文件外，均为 csv 文件，数据均满足，最后一列为标签列，其余为数据列.
            save_path (str, optional): 保存结果的路径. Defaults to "./result/".
            num (int, optional): 聚类类簇数. Defaults to 0.
            dc_percent (int, optional): 截断距离百分比数. Defaults to 1.
            dc_method (int, optional): 截断距离计算方法. Defaults to 1.
            rho_method (int, optional): 局部密度计算方法. Defaults to 0.
            delta_method (int, optional): 相对距离计算方法. Defaults to 0.
            distance_method (str, optional): 度量方式. Defaults to 'ckrod'.
            center (list, optional): 聚类中心，可以人为指定，但为了统一起见，不建议这样做. Defaults to [].
            use_halo (bool, optional): 是否计算光晕点. Defaults to False.
            params (dict, optional): 改进算法需要的其他参数. Defaults to {}.
        """
        super(DpcIRho, self).__init__(
            path,
            save_path,
            num,
            dc_percent,
            dc_method,
            rho_method,
            delta_method,
            distance_method,
            center,
            use_halo,
            params,
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
        """算法名称"""
        self.algorithm_name = "dpc_irho"
        """由于已经创建了以该文件名命名的文件夹，对于文件名只需要添加相关参数"""
        """删去 self.dc_percent，因为在这个类里面，局部密度计算方法不需要该参数"""
        """删去 self.dc_method，因为在这个类里面，无需截断距离"""
        """删去 self.rho_method，因为在这个类里面，局部密度计算方法将做出调整"""
        """度量方法默认为 ckrod"""
        self.file_name = (
            "dem_"
            + str(self.delta_method)
            + "__k_"
            + str(self.params["k"])
            + "__mu_"
            + str(self.params["mu"])
        )
        """use_halo 几乎不用，只有用的时候才加上"""
        if self.use_halo:
            self.file_name += "__ush_" + str(int(self.use_halo))

        return self.distance_ckrod()

    def get_rho(self, dc):
        """
        计算局部密度
        Args:
            dc (_type_): 截断距离，这里用不上
        Returns:
            rho (_type_): 每个样本点的局部密度
        """
        """每个样本点的局部密度"""
        rho = numpy.zeros(self.samples_num)
        """样本权重集合"""
        samples_weight = [0 for _ in range(self.samples_num)]

        """self.dis_matrix 转化为 numpy.array，并按样本距离索引升序"""
        idx_matrix = numpy.array(self.dis_matrix).argsort()[
            :, 1 : int(self.params["k"]) + 1
        ]

        """遍历 k 近邻样本，得到样本权重"""
        for i in idx_matrix:
            for j in i:
                samples_weight[int(j)] += 1.0 / (
                    int(self.params["k"]) * self.samples_num
                )

        """重新计算样本局部密度"""
        for idx in range(self.samples_num):
            for ele in idx_matrix[idx]:
                rho[idx] += (1 - samples_weight[int(ele)]) * self.dis_matrix[idx][
                    int(ele)
                ]
            rho[idx] = int(self.params["k"]) / rho[idx]

        return rho


class DpcIAss(DpcIRho):
    """
    结合了新的距离度量方式与局部密度距离，改进了样本分配策略的 DPC 算法
    Args:
        DpcIRho (_type_): 结合了新的距离度量方式，改进了局部密度距离的 DPC 算法
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
        distance_method="ckrod",
        center=[],
        use_halo=False,
        params={},
    ) -> None:
        """
        初始化相关成员
        Args:
            path (_type_): 文件完整路径，除图片类型文件外，均为 csv 文件，数据均满足，最后一列为标签列，其余为数据列.
            save_path (str, optional): 保存结果的路径. Defaults to "./result/".
            num (int, optional): 聚类类簇数. Defaults to 0.
            dc_percent (int, optional): 截断距离百分比数. Defaults to 1.
            dc_method (int, optional): 截断距离计算方法. Defaults to 1.
            rho_method (int, optional): 局部密度计算方法. Defaults to 0.
            delta_method (int, optional): 相对距离计算方法. Defaults to 0.
            distance_method (str, optional): 度量方式. Defaults to 'ckrod'.
            center (list, optional): 聚类中心，可以人为指定，但为了统一起见，不建议这样做. Defaults to [].
            use_halo (bool, optional): 是否计算光晕点. Defaults to False.
            params (dict, optional): 改进算法需要的其他参数. Defaults to {}.
        """
        super().__init__(
            path,
            save_path,
            num,
            dc_percent,
            dc_method,
            rho_method,
            delta_method,
            distance_method,
            center,
            use_halo,
            params,
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
        """算法名称"""
        self.algorithm_name = "dpc_iass"
        """由于已经创建了以该文件名命名的文件夹，对于文件名只需要添加相关参数"""
        """删去 self.dc_percent，因为在这个类里面，局部密度计算方法不需要该参数"""
        """删去 self.dc_method，因为在这个类里面，无需截断距离"""
        """删去 self.rho_method，因为在这个类里面，局部密度计算方法将做出调整"""
        """度量方法默认为 ckrod"""
        self.file_name = (
            "dem_"
            + str(self.delta_method)
            + "__k_"
            + str(self.params["k"])
            + "__mu_"
            + str(self.params["mu"])
        )
        """use_halo 几乎不用，只有用的时候才加上"""
        if self.use_halo:
            self.file_name += "__ush_" + str(int(self.use_halo))

        return self.distance_ckrod()

    def assign_samples(self, rho, center):
        """
        非聚类中心样本点分配策略
        Args:
            rho (_type_): 局部密度
            center (_type_): 聚类中心样本点，有时候可以通过指定特定的点作为聚类中心，为了统一起见，不推荐这样做，但还是保留了这个选择
        Returns:
            cluster_result (dict(center: str, points: list())): 聚类结果
        """
        """使用一步分配法则预分配标签"""
        tmp_result = self.assign(rho, center)
        """预分配的标签，存放在 self.label_pred"""
        self.gain_label_pred(tmp_result)

        """多步分配策略"""
        self.assign_steps()

        """聚类结果的另一种存放形式--字典"""
        cluster_result = {k: list() for k in set(self.label_pred)}
        for i in range(self.samples_num):
            cluster_result[self.label_pred[i]].append(i)

        return cluster_result

    def assign_steps(self):
        """
        多步分配策略
        """
        """近邻数 k"""
        k = int(self.params["k"])
        """self.dis_matrix 转化为 numpy.array，并按样本距离索引升序"""
        idx_matrix = numpy.array(self.dis_matrix).argsort()[:, 1 : k + 1]
        """标签类型集合"""
        s_set = set(list(range(self.samples_num)))
        h_set = set()
        label_set = set(self.label_pred)

        """第一步分配"""
        label_matrix = numpy.zeros((self.samples_num, k))

        for i in range(self.samples_num):
            for j in range(k):
                """k 近邻样本标签"""
                label_matrix[i, j] = self.label_pred[int(idx_matrix[i, j])]

        """是否存在标签数量大于 k/2 的样本"""
        result_matrix = (
            pandas.DataFrame(label_matrix).apply(pandas.value_counts, axis=1) >= k / 2
        )

        for i in range(self.samples_num):
            for j in label_set:
                if result_matrix.at[i, j]:
                    """存在且标签一致"""
                    if j == self.label_pred[i]:
                        s_set.remove(i)
                        h_set.add(i)

        """第二步分配"""
        for i in s_set:
            """hard label 集合"""
            h_list = [self.label_pred[j] for j in set(idx_matrix[i, :]) & h_set]
            """hard label 过半"""
            if len(h_list) >= k / 2:
                c_dict = Counter(h_list).most_common(1)
                """且存在过半的 label"""
                if c_dict[0][1] > len(h_list) / 2:
                    self.label_pred[i] = c_dict[0][0]


class DpcM(Dpc):
    """
    将三个改进变为可选选项与经典 DPC 算法融合
    Args:
        Dpc (_type_): DPC 算法基类，也可以从 DpcD 继承
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
        center=...,
        use_halo=False,
        params={},
    ) -> None:
        """
        初始化相关成员
        Args:
            path (_type_): 文件完整路径，除图片类型文件外，均为 csv 文件，数据均满足，最后一列为标签列，其余为数据列.
            save_path (str, optional): 保存结果的路径. Defaults to "./result/".
            num (int, optional): 聚类类簇数. Defaults to 0.
            dc_percent (int, optional): 截断距离百分比数. Defaults to 1.
            dc_method (int, optional): 截断距离计算方法. Defaults to 1.
            rho_method (int, optional): 局部密度计算方法. Defaults to 0.
            delta_method (int, optional): 相对距离计算方法. Defaults to 0.
            distance_method (str, optional): 度量方式. Defaults to 'ckrod'.
            center (list, optional): 聚类中心，可以人为指定，但为了统一起见，不建议这样做. Defaults to [].
            use_halo (bool, optional): 是否计算光晕点. Defaults to False.
            params (dict, optional): 改进算法需要的其他参数，将之前的 assign_method 加入到 params 中. Defaults to {}.
        """
        super().__init__(
            path,
            save_path,
            num,
            dc_percent,
            dc_method,
            rho_method,
            delta_method,
            distance_method,
            center,
            use_halo,
            params,
        )

    def load_points_msg(self):
        """
        相较于之前的，多了一些的选项，这些选项一方面对应着不同的方法(经典的与改进)，一方面对应着文件名
        获取数据集相关信息，如距离矩阵，欧式距离列表，最大距离，最小距离
        Args:
        Returns:
            dis_array (_type_): 样本间距离的矩阵
            max_dis (_type_): 样本间最大距离
            min_dis (_type_): 样本间最小距离
        """
        """算法名称"""
        self.algorithm_name = "dpc_" + self.distance_method

        """判断度量方法，决定百分比数，截断距离方法"""
        if self.distance_method in METRIC_METHOD:
            """使用 sch.distance.pdist 中提供的方法"""
            self.file_name = (
                "dcp_" + str(self.dc_percent) + "__dcm_" + str(self.dc_method)
            )

            return self.distance_standard(self.distance_method)
        elif self.distance_method in {"rod", "krod", "ckrod"}:
            if self.distance_method == "krod":
                """krod"""
                self.file_name = (
                    "dcp_"
                    + str(self.dc_percent)
                    + "__mu_"
                    + str(self.params["mu"])
                    + "__dcm_"
                    + str(self.dc_method)
                )
            elif self.distance_method == "ckrod":
                """ckrod 需要参数 k，mu，不需要百分比数，不需要截断距离方法"""
                self.file_name = (
                    "k_" + str(self.params["k"]) + "__mu_" + str(self.params["mu"])
                )
            else:
                """rod"""
                self.file_name = (
                    "dcp_" + str(self.dc_percent) + "__dcm_" + str(self.dc_method)
                )

            """rod 系列度量方法"""
            return self.distance_rods()

        """判断局部密度"""
        if self.rho_method == 3:
            """改进了局部密度"""
            self.algorithm_name += "_irho"
        else:
            """经典的方法"""
            self.algorithm_name += "_Rho"
        self.file_name += "__rhom_" + str(self.rho_method)

        """加上相对距离"""
        self.file_name += "__dem_" + str(self.delta_method)

        """加上分配策略"""
        if "assign_method" in self.params.keys():
            if self.params["assign_method"] == 0:
                """一步分配策略"""
                self.algorithm_name += "_ass"
            elif self.params["assign_method"] == 1:
                """两步分配策略"""
                self.algorithm_name += "_iass"
        else:
            """默认使用一步分配策略"""
            self.params["assign_method"] == 0
            self.algorithm_name += "_ass"

        """use_halo 几乎不用，只有用的时候才加上"""
        if self.use_halo:
            self.file_name += "__ush_" + str(int(self.use_halo))

    def distance_rods(self):
        """
        ROD 系列度量方式
        Returns:
            dis_array (_type_): 样本间距离的矩阵
            max_dis (_type_): 样本间最大距离
            min_dis (_type_): 样本间最小距离
        """
        """先根据欧式距离生成所有样本的距离列表"""
        dis_array = sch.distance.pdist(self.samples, "euclidean")
        """这里采用的方法是先深拷贝一份 self.dis_matrix"""
        euclidean_table = self.dis_matrix.copy()

        """处理距离矩阵"""
        num = 0
        for i in range(self.samples_num):
            for j in range(i + 1, self.samples_num):
                """赋值"""
                self.dis_matrix.at[i, j] = dis_array[num]
                """处理对角元素"""
                self.dis_matrix.at[j, i] = self.dis_matrix.at[i, j]
                num += 1

        """对 euclidean_table 使用 argsort()，该函数会对矩阵的每一行从小到大排序，返回的是 euclidean_table 中索引"""
        rank_order_table = numpy.array(euclidean_table).argsort()

        """优化下度量计算"""
        """用新的度量方式得到的样本间距离覆盖掉 dis_array"""
        dis_array = [
            self.rods_fun(i, j, rank_order_table, euclidean_table)
            for i in range(self.samples_num)
            for j in range(i + 1, self.samples_num)
        ]

        """对距离矩阵进行处理"""
        num = 0
        for i in range(self.samples_num):
            for j in range(i + 1, self.samples_num):
                """self.dis_matrix 内存放样本间的距离"""
                self.dis_matrix.at[i, j] = dis_array[num]
                """处理对角元素"""
                self.dis_matrix.at[j, i] = self.dis_matrix.at[i, j]
                num += 1

        """最大距离"""
        max_dis = self.dis_matrix.max().max()
        """最小距离"""
        min_dis = self.dis_matrix.min().min()

        return dis_array, max_dis, min_dis

    def rods_fun(self, i, j, rank_order_table, euclidean_table):
        """
        rod 及其改进算法
        Args:
            i (_type_): 第 i 个样本
            j (_type_): 第 j 个样本
            rank_order_table (_type_): 排序距离表
            euclidean_table (_type_): 欧式距离表
        Returns:
            dis (_type_): i 与 j 之间的距离
        """
        """xi 与 xj 之间的距离"""
        dis = -1
        """第 i 个样本"""
        x1 = rank_order_table[i, :]
        """第 j 个样本"""
        x2 = rank_order_table[j, :]
        """rod 及其改进算法需要的一些前置条件"""
        """x1 样本的 id"""
        id_x1 = x1[0]
        """x2 样本的 id"""
        id_x2 = x2[0]
        """o_a_b，b 在 a 的排序列表中的索引位置"""
        o_a_b = numpy.where(x1 == id_x2)[0][0]
        """o_b_a，a 在 b 的排序列表中的索引位置"""
        o_b_a = numpy.where(x2 == id_x1)[0][0]

        """判断度量方法"""
        if self.distance_method == "rod":
            """d_a_b，在 a 的排序列表中，从 a 到 b 之间的所有元素在 b 的排序列表中的序数或者索引之和"""
            """先切片，a 的排序列表中 [a, b] 的索引列表"""
            slice_a_b = x1[0 : o_a_b + 1]
            """索引切片在 b 中的位置序数之和"""
            d_a_b = sum(numpy.where(x2 == slice_a_b[:, None])[-1])
            """d_b_a，在 b 的排序列表中，从 b 到 a 之间的所有元素在 a 的排序列表中的序数或者索引之和"""
            """先切片，b 的排序列表中 [b, a] 的索引列表"""
            slice_b_a = x2[0 : o_b_a + 1]
            """索引切片在 a 中的位置序数之和"""
            d_b_a = sum(numpy.where(x1 == slice_b_a[:, None])[-1])
            """rod"""
            dis = (d_a_b + d_b_a) / min(o_a_b, o_b_a)
        elif self.distance_method == "krod":
            """改进 rod"""
            l_a_b = o_a_b + o_b_a
            """高斯核"""
            k_a_b = math.exp(-((euclidean_table.at[i, j] / self.params["mu"]) ** 2))
            """krod，改进 rod 与高斯核相结合"""
            dis = l_a_b / k_a_b
        elif self.distance_method == "ckrod":
            """改进 krod"""
            l_a_b = (o_a_b + o_b_a) / (self.samples_num - 1) / self.features_num
            """高斯核"""
            k_a_b = math.exp(-((euclidean_table.at[i, j] / self.params["mu"]) ** 2))
            """ckrod，改进 krod 与高斯核相结合"""
            dis = l_a_b / k_a_b

        return dis

    def get_rho(self, dc):
        """
        计算局部密度
        Args:
            dc (_type_): 截断距离，这里用不上
        Returns:
            rho (_type_): 每个样本点的局部密度
        """
        """每个样本点的局部密度"""
        rho = numpy.zeros(self.samples_num)

        """判断局部密度计算方法"""
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
            """样本权重集合"""
            samples_weight = [0 for _ in range(self.samples_num)]

            """self.dis_matrix 转化为 numpy.array，并按样本距离索引升序"""
            idx_matrix = numpy.array(self.dis_matrix).argsort()[
                :, 1 : int(self.params["k"]) + 1
            ]

            """遍历 k 近邻样本，得到样本权重"""
            for i in idx_matrix:
                for j in i:
                    samples_weight[int(j)] += 1.0 / (
                        int(self.params["k"]) * self.samples_num
                    )

            """重新计算样本局部密度"""
            for idx in range(self.samples_num):
                for ele in idx_matrix[idx]:
                    rho[idx] += (1 - samples_weight[int(ele)]) * self.dis_matrix[idx][
                        int(ele)
                    ]
                rho[idx] = int(self.params["k"]) / rho[idx]

        return rho

    def assign_samples(self, rho, center):
        """
        非聚类中心样本点分配策略
        Args:
            rho (_type_): 局部密度
            center (_type_): 聚类中心样本点，有时候可以通过指定特定的点作为聚类中心，为了统一起见，不推荐这样做，但还是保留了这个选择
        Returns:
            cluster_result (dict(center: str, points: list())): 聚类结果
        """
        """判断样本分配策略"""
        if self.params["assign_method"] == 0:
            """使用一步分配法则分配标签"""
            return self.assign(rho, center)
        elif self.params["assign_method"] == 1:
            """使用一步分配法则预分配标签"""
            tmp_result = self.assign(rho, center)
            """预分配的标签，存放在 self.label_pred"""
            self.gain_label_pred(tmp_result)

            """多步分配策略"""
            self.assign_steps()

            """聚类结果的另一种存放形式--字典"""
            cluster_result = {k: list() for k in set(self.label_pred)}
            for i in range(self.samples_num):
                cluster_result[self.label_pred[i]].append(i)

            return cluster_result

    def assign_steps(self):
        """
        多步分配策略
        """
        """近邻数 k"""
        k = int(self.params["k"])
        """self.dis_matrix 转化为 numpy.array，并按样本距离索引升序"""
        idx_matrix = numpy.array(self.dis_matrix).argsort()[:, 1 : k + 1]
        """标签类型集合"""
        s_set = set(list(range(self.samples_num)))
        h_set = set()
        label_set = set(self.label_pred)

        """第一步分配"""
        label_matrix = numpy.zeros((self.samples_num, k))

        for i in range(self.samples_num):
            for j in range(k):
                """k 近邻样本标签"""
                label_matrix[i, j] = self.label_pred[int(idx_matrix[i, j])]

        """是否存在标签数量大于 k/2 的样本"""
        result_matrix = (
            pandas.DataFrame(label_matrix).apply(pandas.value_counts, axis=1) >= k / 2
        )

        for i in range(self.samples_num):
            for j in label_set:
                if result_matrix.at[i, j]:
                    """存在且标签一致"""
                    if j == self.label_pred[i]:
                        s_set.remove(i)
                        h_set.add(i)

        """第二步分配"""
        for i in s_set:
            """hard label 集合"""
            h_list = [self.label_pred[j] for j in set(idx_matrix[i, :]) & h_set]
            """hard label 过半"""
            if len(h_list) >= k / 2:
                c_dict = Counter(h_list).most_common(1)
                """且存在过半的 label"""
                if c_dict[0][1] > len(h_list) / 2:
                    self.label_pred[i] = c_dict[0][0]
