# -*- coding: utf-8 -*-


from codes.algorithm import *
from .dataSetting import *
from multiprocessing import Pool

conf = dict()
"""Demo 中使用到的度量方法，这里用简写"""
DIS_METHOD = ["cosine", "euc", "gau", "man", "rod", "krod", "ckrod"]

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
