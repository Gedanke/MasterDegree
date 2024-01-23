# -*- coding: utf-8 -*-


from codes.algorithm import *
from .dataSetting import *
from multiprocessing import Pool

conf = dict()
"""Demo 中使用到的度量方法，这里用简写"""
MOONS_DIS_METHOD = ["euc", "man", "gau", "rod", "krod", "ckrod"]
# MOONS_DIS_METHOD = ["rod"]

"""
运行算法
"""
ALGORITHM_LIST = [
    "AC",
    "AP",
    "Birch",
    "Dbscan",
    "Kmeans",
    "MeanShit",
    "Optics",
    "Sc",
    "Dpc",
    "DpcD",
    "DpcKnn",
    "SnnDpc",
    "DpcCkrod",
    "DpcIRho",
    "DpcIAss",
]


class RunDemo:
    """
    在 demo 数据集上进行实验
    使用一些度量方式，运行 demo 数据集(双月，双圈)，获得实验结果
    """

    def __init__(self, path="./dataset/experiment/demo/", params={}) -> None:
        """
        初始化相关成员
        Args:
            path (str, optional): 文件路径，由于调用该类的主程序路径待定，这里使用手动传入，同时根据该路径解析出保存路径. Defaults to "./dataset/experiment/demo/".
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
        self.label_true = self.samples[col[-1]]
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
                    self.label_true,
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
            os.mkdir(self.save_path + "result/circles/")

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
                        self.label_true,
                        dis_name,
                        file_path + dis_name + ".csv",
                    ),
                )

        p.close()
        p.join()


def run_data_algorithm(pool, dataset_params, algorithm_list, distance_method):
    """
    在 dataset 上运行 algorithm
    Args:
        pool (_type_): 进程池
        dataset_params (_type_): 指定数据集
        algorithm_list (_type_): 指定算法
        algorithm_list (_type_): 度量方法
    """
    """数据集相关的基本信息"""
    path = dataset_params["path"]
    save_path = dataset_params["save_path"]
    num = dataset_params["num"]

    if "AC" in algorithm_list:
        pool.apply_async(
            ComAC,
            args=(
                path,
                save_path,
                num,
                ALGORITHM_LIST["AC"],
            ),
        )
    if "AP" in algorithm_list:
        pool.apply_async(
            ComAP,
            args=(
                path,
                save_path,
                num,
                ALGORITHM_LIST["AP"],
            ),
        )
    if "Birch" in algorithm_list:
        for _threshold in ALGORITHM_PARAMS["Birch"]["threshold"]:
            pool.apply_async(
                ComBirch,
                args=(
                    path,
                    save_path,
                    num,
                    {"threshold": _threshold},
                ),
            )
    if "Dbscan" in algorithm_list:
        for _eps in ALGORITHM_PARAMS["Dbscan"]["eps"]:
            for _min_samples in ALGORITHM_PARAMS["Dbscan"]["min_samples"]:
                pool.apply_async(
                    ComDBSCAN,
                    args=(
                        path,
                        save_path,
                        num,
                        {"eps": _eps, "min_samples": _min_samples},
                    ),
                )
    if "Kmeans" in algorithm_list:
        pool.apply_async(
            ComKMeans,
            args=(
                path,
                save_path,
                num,
                ALGORITHM_LIST["Kmeans"],
            ),
        )
    if "MeanShit" in algorithm_list:
        pool.apply_async(
            ComMeanShit,
            args=(
                path,
                save_path,
                num,
                ALGORITHM_LIST["MeanShit"],
            ),
        )
    if "Optics" in algorithm_list:
        for _eps in ALGORITHM_PARAMS["Optics"]["eps"]:
            for _min_samples in ALGORITHM_PARAMS["Optics"]["min_samples"]:
                pool.apply_async(
                    ComOPTICS,
                    args=(
                        path,
                        save_path,
                        num,
                        {"eps": _eps, "min_samples": _min_samples},
                    ),
                )
    if "Sc" in algorithm_list:
        for _gamma in ALGORITHM_PARAMS["Sc"]["gamma"]:
            pool.apply_async(
                ComSC,
                args=(
                    path,
                    save_path,
                    num,
                    {"gamma": _gamma},
                ),
            )
    if "Dpc" in algorithm_list:
        for _percent in ALGORITHM_PARAMS["Dpc"]["percent"]:
            pool.apply_async(
                Dpc,
                args=(
                    path,
                    save_path,
                    num,
                    _percent,
                ),
            )
    if "DpcD" in algorithm_list:
        if distance_method == "krod":
            """krod"""
            for _percent in ALGORITHM_PARAMS["DpcD"]["percent"]:
                for _mu in ALGORITHM_PARAMS["DpcD"]["mu"]:
                    pool.apply_async(
                        DpcD,
                        args=(
                            path,
                            save_path,
                            num,
                            _percent,
                            1,
                            0,
                            0,
                            distance_method,
                            [],
                            False,
                            {"mu": _mu},
                        ),
                    )
        else:
            """这里使用 rod，其他距离也是沿用该方法"""
            for _percent in ALGORITHM_PARAMS["DpcD"]["percent"]:
                pool.apply_async(
                    DpcD,
                    args=(path, save_path, num, _percent, 1, 0, 0, distance_method),
                )
    if "DpcKnn" in algorithm_list:
        for _percent in ALGORITHM_PARAMS["DpcKnn"]["percent"]:
            pool.apply_async(
                DpcKnn,
                args=(
                    path,
                    save_path,
                    num,
                    _percent,
                ),
            )
    if "SnnDpc" in algorithm_list:
        for _k in ALGORITHM_PARAMS["SnnDpc"]["k"]:
            pool.apply_async(
                SnnDpc,
                args=(
                    path,
                    save_path,
                    num,
                    _k,
                ),
            )
    if "DpcCkrod" in algorithm_list:
        for _percent in ALGORITHM_PARAMS["DpcCkrod"]["percent"]:
            for _mu in ALGORITHM_PARAMS["DpcCkrod"]["mu"]:
                pool.apply_async(
                    DpcCkrod,
                    args=(
                        path,
                        save_path,
                        num,
                        _percent,
                        1,
                        0,
                        0,
                        "ckrod",
                        [],
                        False,
                        {"mu": _mu},
                    ),
                )
    if "DpcIRho" in algorithm_list:
        for _k in ALGORITHM_PARAMS["DpcIRho"]["k"]:
            for _mu in ALGORITHM_PARAMS["DpcIRho"]["mu"]:
                pool.apply_async(
                    DpcIRho,
                    args=(
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
                        {"k": _k, "mu": _mu},
                    ),
                )
    if "DpcIAss" in algorithm_list:
        for _k in ALGORITHM_PARAMS["DpcIAss"]["k"]:
            for _mu in ALGORITHM_PARAMS["DpcIAss"]["mu"]:
                pool.apply_async(
                    DpcIAss,
                    args=(
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
                        {"k": _k, "mu": _mu},
                    ),
                )


class RunSynthesis:
    """
    将 ALGORITHM_PARAMS 与 SYNTHESIS_PARAMS 合并，运行 synthesis 数据集
    """

    def __init__(
        self, path="./dataset/experiment/synthesis/", dataset_list=[], algorithm_list=[]
    ) -> None:
        """
        初始化相关成员
        Args:
            path (str, optional): 文件路径，由于调用该类的主程序路径待定，这里使用手动传入，同时根据该路径解析出保存路径. Defaults to "./dataset/experiment/synthesis/".
            dataset_list (list, optional): 使用的数据集列表. Defaults to [].
            algorithm_list (list, optional): 使用的算法列表. Defaults to [].
        """
        """文件路径"""
        self.path = path
        """保存结果路径"""
        self.save_path = self.path.replace("/experiment", "").replace(
            "dataset", "result"
        )
        """数据集列表，为空，使用全部的数据集"""
        self.dataset_list = dataset_list
        if dataset_list == []:
            self.dataset_list = list(SYNTHESIS_PARAMS.keys())
        """使用的算法列表，为空，使用全部的算法(除了 DpcM。选项太多)"""
        if algorithm_list == []:
            self.algorithm_list = ALGORITHM_LIST

    def deal_synthesis(self):
        """
        处理 synthesis 数据集
        """
        for data_name in self.dataset_list:
            """遍历数据集"""
        

class RunUci:
    """
    将 ALGORITHM_PARAMS 与 SYNTHESIS_PARAMS 合并，运行 uci 数据集
    """


class RunImage:
    """
    将 ALGORITHM_PARAMS 与 IMAGE_PARAMS 合并，运行 image 数据集
    """
