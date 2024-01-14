# -*- coding: utf-8 -*-


from codes.algorithm import *
from .dataSetting import *
from multiprocessing.pool import Pool

"""
运行算法
"""

conf = dict()


def test_dataset(data_type):
    """
    统计数据集信息
    Args:
        data_type (_type_): _description_
    """
    configure = {}
    path = "./dataset/experiment/" + data_type + "/"
    for data_dir in os.listdir(path):
        configure[data_dir] = {}
        p = path + data_dir + "/"
        file = os.listdir(p)[0]
        data = pandas.read_csv(p + file)
        cols = data.columns
        configure[data_dir]["path"] = p + file
        configure[data_dir]["save_path"] = "./result/" + data_type + "/"
        configure[data_dir]["samples_num"] = len(data)
        configure[data_dir]["features_num"] = len(cols) - 1
        configure[data_dir]["num"] = len(set(data[cols[-1]]))

    print(configure)
