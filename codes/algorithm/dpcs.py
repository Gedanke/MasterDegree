# -*- coding: utf-8 -*-


from .dpc import *
from numpy import (
    arange,
    argsort,
    argwhere,
    empty,
    full,
    inf,
    intersect1d,
    max,
    sort,
    sum,
    zeros,
)
from scipy.spatial.distance import pdist, squareform

"""
DPC 相关的算法
"""

"""sch.distance.pdist 中提供的方法"""
METRIC_METHOD = {
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "euclidean",
    "hamming",
    "jaccard",
    "jensenshannon",
    "kulczynski1",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
}
"""rod 系列度量方法"""
ROD_METHOD = {"rod", "krod"}


class DpcD(Dpc):
    """
    可以选择不同种度量方式的 DPC 算法，不包含 ckrod
    Args:
        Dpc (_type_): DPC 算法基类
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
        初始化相关成员
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
        super(DpcD, self).__init__(
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
        self.algorithm_name = "dpc_" + self.distance_method
        """文件名"""
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

        """判断度量方法"""
        if self.distance_method in METRIC_METHOD:
            """使用 sch.distance.pdist 中提供的方法"""
            return self.distance_standard(self.distance_method)
        elif self.distance_method in ROD_METHOD:
            if self.distance_method == "krod":
                """krod 需要参数 mu"""
                self.file_name += "__mu_" + str(self.params["mu"])

            """rod 系列度量方法"""
            return self.distance_rods()

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
                euclidean_table.at[i, j] = dis_array[num]
                """处理对角元素"""
                euclidean_table.at[j, i] = euclidean_table.at[i, j]
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
            d_a_b = int(sum(numpy.where(x2 == slice_a_b[:, None])[-1]))
            """d_b_a，在 b 的排序列表中，从 b 到 a 之间的所有元素在 a 的排序列表中的序数或者索引之和"""
            """先切片，b 的排序列表中 [b, a] 的索引列表"""
            slice_b_a = x2[0 : o_b_a + 1]
            """索引切片在 a 中的位置序数之和"""
            d_b_a = int(sum(numpy.where(x1 == slice_b_a[:, None])[-1]))
            """rod"""
            if o_a_b == 0 and o_b_a == 0:
                """相同的样本，距离为 0"""
                dis = 0
            else:
                dis = (d_a_b + d_b_a) / min(o_a_b, o_b_a)
        elif self.distance_method == "krod":
            """krod"""
            l_a_b = o_a_b + o_b_a
            """高斯核"""
            k_a_b = math.exp(-((euclidean_table.at[i, j] / self.params["mu"]) ** 2))
            """krod，改进 rod 与高斯核相结合"""
            dis = l_a_b / k_a_b

        return dis


class DpcKnn(Dpc):
    """
    DPC-KNN 算法
    Args:
        Dpc (_type_): DPC 算法基类
    """

    def __init__(
        self,
        path,
        save_path="./result/",
        num=0,
        dc_percent=1,
        dc_method=1,
        rho_method=3,
        delta_method=1,
        distance_method="euclidean",
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
            rho_method (int, optional): 局部密度计算方法. Defaults to 3.
            delta_method (int, optional): 相对距离计算方法. Defaults to 1.
            distance_method (str, optional): 度量方式. Defaults to 'euclidean'.
            center (list, optional): 聚类中心，可以人为指定，但为了统一起见，不建议这样做. Defaults to [].
            use_halo (bool, optional): 是否计算光晕点. Defaults to False.
            params (dict, optional): 改进算法需要的其他参数. Defaults to {}.
        """
        super(DpcKnn, self).__init__(
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
        if self.distance_method in METRIC_METHOD:
            """算法名称"""
            self.algorithm_name = "dpc_knn"
            """文件名"""
            """由于已经创建了以该文件名命名的文件夹，对于文件名只需要添加相关参数"""
            """删去 self.dc_method，因为在这个类里面，截断距离都是通过 self.dc_percent 指定的"""
            """删去 self.rho_method，因为在这个类里面，局部密度方式默认为 3，即 重写了 get_rho"""
            """删去 self.distance_method，因为在这个类里面，度量方式是默认的"""
            self.file_name = (
                "dcp_" + str(self.dc_percent) + "__dem_" + str(self.delta_method)
            )
            """use_halo 几乎不用，只有用的时候才加上"""
            if self.use_halo:
                self.file_name += "__ush_" + str(int(self.use_halo))

            return self.distance_standard()

    def get_rho(self, dc):
        """
        改进局部密度计算方式
        Args:
            dc (_type_): 截断距离
        Returns:
            rho (_type_): 每个样本点的局部密度
        """
        """每个样本点的局部密度"""
        rho = numpy.zeros(self.samples_num)
        """近邻样本数"""
        k = int(self.samples_num * self.dc_percent / 100)
        """实验参数设置中尽量不要把 dc_percent 设置得太小"""
        if k <= 0:
            return rho

        """self.dis_matrix 转化为 numpy.array"""
        matrix = numpy.array(self.dis_matrix)
        """按样本距离排序，这里按行排序"""
        matrix.sort()
        """k 近邻个样本，1 到 1 + self.params["k"]"""
        exp = -(matrix[:, 1 : 1 + int(k)] ** 2).sum(axis=1) / int(k)

        """局部密度"""
        for i in range(self.samples_num):
            rho[i] = math.exp(exp[i])

        return rho


class DpcFKnn(Dpc):
    """
    FKNN-DPC 算法
    Args:
        Dpc (_type_): DPC 算法基类
    """

    def __init__(
        self,
        path,
        save_path="./result/",
        num=0,
        dc_percent=1,
        dc_method=1,
        rho_method=3,
        delta_method=1,
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
            rho_method (int, optional): 局部密度计算方法. Defaults to 3.
            delta_method (int, optional): 相对距离计算方法. Defaults to 1.
            distance_method (str, optional): 度量方式. Defaults to 'euclidean'.
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


class SnnDpc:
    """
    SNN-DPC 算法
    """

    def __init__(self, path, save_path="./result/", num=0, k=3):
        """
        初始化相关成员
        Args:
            path (_type_): 文件完整路径，除图片类型文件外，均为 csv 文件，数据均满足，最后一列为标签列，其余为数据列.
            save_path (str, optional): 保存结果的路径. Defaults to "./result/".
            num (int, optional): 聚类类簇数. Defaults to 0.
            k (int, optional): 近邻数. Defaults to 3.
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
        """近邻数"""
        self.k = int(k)

        """其他参数"""
        """数据集的所有样本点，不包括标签列"""
        self.samples = pandas.DataFrame({})
        """样本个数"""
        self.samples_num = 0
        """真实标签列"""
        self.label_true = list()
        """聚类结果"""
        self.label_pred = list()
        """聚类结果指标"""
        self.cluster_result = dict()
        """算法名称"""
        self.algorithm_name = "snn_dpc"
        """文件名"""
        self.file_name = "snn_dpc__k_" + str(self.k)

        """聚类函数"""
        self.cluster()

    def cluster(self):
        """
        聚类算法
        """
        """加载数据集"""
        self.init_points_msg()
        """SNN-DPC"""
        self.center, self.label_pred = self.snn_dpc()
        """转化为 numpy"""
        self.label_pred = self.label_pred.tolist()
        """聚类结果，指标写入到 json 文件中"""
        self.save_result()

    def init_points_msg(self):
        """
        完成部分固定信息的初始化
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
        self.samples = numpy.array(self.samples.loc[:, cols[0:-1]])

    def snn_dpc(self):
        """
        SNN-DPC 算法，基于作者提供的代码，并未做太多修改
        Returns:
            index_assignment (_type_): 聚类结果
        """
        unassigned = -1
        n, d = self.samples.shape

        # Compute distance
        distance = squareform(pdist(self.samples))

        # Compute neighbor
        index_distance_asc = argsort(distance)
        index_neighbor = index_distance_asc[:, : self.k]

        # Compute shared neighbor
        index_shared_neighbor = empty([n, n, self.k], int)
        num_shared_neighbor = empty([n, n], int)
        for i in range(n):
            num_shared_neighbor[i, i] = 0
            for j in range(i):
                shared = intersect1d(
                    index_neighbor[i], index_neighbor[j], assume_unique=True
                )
                num_shared_neighbor[j, i] = num_shared_neighbor[i, j] = shared.size
                index_shared_neighbor[j, i, : shared.size] = index_shared_neighbor[
                    i, j, : shared.size
                ] = shared

        # Compute similarity
        # Diagonal and some elements are 0
        similarity = zeros([n, n])
        for i in range(n):
            for j in range(i):
                if (
                    i in index_shared_neighbor[i, j]
                    and j in index_shared_neighbor[i, j]
                ):
                    index_shared = index_shared_neighbor[
                        i, j, : num_shared_neighbor[i, j]
                    ]
                    distance_sum = sum(
                        distance[i, index_shared] + distance[j, index_shared]
                    )
                    similarity[i, j] = similarity[j, i] = (
                        num_shared_neighbor[i, j] ** 2 / distance_sum
                    )

        # Compute ρ
        rho = sum(sort(similarity)[:, -self.k :], axis=1)

        # Compute δ
        distance_neighbor_sum = empty(n)
        for i in range(n):
            distance_neighbor_sum[i] = sum(distance[i, index_neighbor[i]])
        index_rho_desc = argsort(rho)[::-1]
        delta = full(n, inf)
        for i, a in enumerate(index_rho_desc[1:], 1):
            for b in index_rho_desc[:i]:
                if delta[a] < distance[a, b] * (
                    distance_neighbor_sum[a] + distance_neighbor_sum[b]
                ):
                    delta[a] = delta[a]
                else:
                    delta[a] = distance[a, b] * (
                        distance_neighbor_sum[a] + distance_neighbor_sum[b]
                    )
        delta[index_rho_desc[0]] = -inf
        delta[index_rho_desc[0]] = max(delta)

        # Compute γ
        gamma = rho * delta

        # Compute centroid
        index_assignment = full(n, unassigned)
        index_centroid = sort(argsort(gamma)[-self.num :])
        # index_centroid = numpy.array([12, 253])
        index_assignment[index_centroid] = arange(self.num)

        # Assign non-centroid step 1
        queue = index_centroid.tolist()
        while queue:
            a = queue.pop(0)
            for b in index_neighbor[a]:
                if (
                    index_assignment[b] == unassigned
                    and num_shared_neighbor[a, b] >= self.k / 2
                ):
                    index_assignment[b] = index_assignment[a]
                    queue.append(b)

        # Assign non-centroid step 2
        index_unassigned = argwhere(index_assignment == unassigned).flatten()
        while index_unassigned.size:
            num_neighbor_assignment = zeros([index_unassigned.size, self.num], int)
            for i, a in enumerate(index_unassigned):
                for b in index_distance_asc[a, : self.k]:
                    if index_assignment[b] != unassigned:
                        num_neighbor_assignment[i, index_assignment[b]] += 1
            most = max(num_neighbor_assignment)
            if most:
                temp = argwhere(num_neighbor_assignment == most)
                index_assignment[index_unassigned[temp[:, 0]]] = temp[:, 1]
                index_unassigned = argwhere(index_assignment == unassigned).flatten()
            else:
                self.k += 1

        return index_centroid, index_assignment

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
        """保存聚类中心"""
        self.cluster_result["center"] = self.center.tolist()
        """只保存标签列"""
        self.cluster_result["label"] = self.label_pred
        """合并聚类指标"""
        self.cluster_result.update(
            cluster_index(self.samples, self.label_true, self.label_pred, self.num != 0)
        )
        """写入文件"""
        with open(self.get_file_path(), "w", encoding="utf-8") as f:
            f.write(json.dumps(self.cluster_result, ensure_ascii=False))
