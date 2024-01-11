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
ROD_METHOD = {"rod", "krod", "ckrod"}


class DpcD(Dpc):
    """
    可以选择不同种度量方式的 DPC 算法，不包含 ckrod
    Args:
        Dpc (_type_): DPC 算法基类
    """


class DpcKnn(Dpc):
    """
    DPC-KNN 算法
    Args:
        Dpc (_type_): DPC 算法基类
    """


class DpcFKnn(Dpc):
    """
    FKNN-DPC 算法
    Args:
        Dpc (_type_): DPC 算法基类
    """


class SnnDpc:
    """
    SNN-DPC 算法
    """
