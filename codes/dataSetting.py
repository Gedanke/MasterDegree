# -*- coding: utf-8 -*-


from sklearn.datasets import *


def normalized_label(label: list) -> list:
    """
    将标签转换为从 0 开始的整数，这里采用的方法是取最小的数，将所有的数据减去最小的数据
    Args:
        label (list): 原有标签
    Returns:
        list: 规整后的标签
    """
    """最小的数据"""
    min_value = min(label)
    """规整后的标签"""
    new_label = [int(value - min_value) for value in label]

    return new_label


def get_moons(params):
    """
    https://blog.csdn.net/chenxy_bwave/article/details/122078564
    生成并返回双月数据集
    Returns:
        _type_: 数据集以及标签
    """
    if "noise" in params.keys():
        data, label = make_moons(
            n_samples=params["num"],
            shuffle=True,
            noise=params["noise"],
            random_state=1,
        )
    else:
        """无噪声"""
        data, label = make_moons(
            n_samples=params["num"],
            shuffle=True,
            noise=0,
            random_state=1,
        )

    return data, label


def get_circles(params):
    """
    https://blog.csdn.net/chenxy_bwave/article/details/122078564
    生成并返回双月数据集
    Returns:
        _type_: 数据集以及标签
    """
    if "noise" in params.keys():
        data, label = make_circles(
            n_samples=params["num"],
            shuffle=True,
            noise=params["noise"],
            random_state=1,
            factor=params["factor"],
        )
    else:
        """无噪声"""
        data, label = make_circles(
            n_samples=params["num"],
            shuffle=True,
            noise=0,
            random_state=1,
            factor=params["factor"],
        )

    return data, label
