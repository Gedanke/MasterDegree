# -*- coding: utf-8 -*-


from codes.algorithm import *
from .dataSetting import *
from multiprocessing.pool import Pool

"""
运行算法
"""


def test_dataset(data_type):
    """
    统计数据集信息
    Args:
        data_type (_type_): _description_
    """
    path = "./dataset/experiment/" + data_type
    for data in os.listdir(path):
        print(data)
