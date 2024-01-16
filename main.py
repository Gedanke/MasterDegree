# -*- coding: utf-8 -*-


# import sys
# import os

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)

from codes import *


def test_process():
    """
    test for process
    """
    param = {
        "norm": 0,
        "gmu": 1,
        "sigma": 1,
        "noise": 0.1,
        "type": "moons",
        "noise_type": 0,
        "num": 128,
        "mu": 4,
        "factor": 0.5,
    }
    # dd = DealData("demo", param)
    # dd.deal_demo()
    # dd.get_demo()
    dd = DealData("synthesis", param)
    dd.deal_synthesis()
    dd.get_synthesis()
    dd = DealData("uci", param)
    dd.deal_uci()
    dd.get_uci()


def test_compare():
    """_summary_"""
    ck = ComKMeans(
        "./dataset/experiment/synthesis/aggregation/aggregation.csv",
        "./result/synthesis/",
        7,
        {},
    )
    ck.cluster()
    cap = ComAP(
        "./dataset/experiment/synthesis/aggregation/aggregation.csv",
        "./result/synthesis/",
        7,
        {},
    )
    cap.cluster()


def test_dpc():
    """ """
    dpc = Dpc(
        "./dataset/experiment/synthesis/spiral/spiral.csv",
        "./result/synthesis/",
        3,
        1.8,
        1,
        0,
        0,
        "euclidean",
        [],
        False,
    )
    dpc.cluster()


def test_datasets():
    """
    统计数据集信息
    """
    test_dataset("synthesis")
    test_dataset("uci")


if __name__ == "__main__":
    """"""
    # test_process()
    # test_compare()
    # test_dpc()
    test_datasets()