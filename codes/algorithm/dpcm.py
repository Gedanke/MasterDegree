# -*- coding: utf-8 -*-


from .dpcs import *
from collections import Counter


"""
论文中涉及到三个 DPC 改进算法
"""


class DpcCkrod(Dpc):
    """
    改进了距离度量的 dpc 算法
    Args:
        Dpc (_type_): DPC 算法基类
    """


class DpcIRho(DpcCkrod):
    """
    结合了新的距离度量方式，改进了局部密度距离的 dpc 算法
    Args:
        DpcCkrod (_type_): 改进了距离度量的 DPC 算法
    """


class DpcIAss(DpcIRho):
    """
    结合了新的距离度量方式与局部密度距离，改进了样本分配策略的 dpc 算法
    Args:
        DpcIRho (_type_): 结合了新的距离度量方式，改进了局部密度距离的 dpc 算法
    """
