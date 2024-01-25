# -*- coding: utf-8 -*-


import sys
import os

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)

from codes import *

params = {
    "moons": {
        "norm": 0,
        "gmu": 1,
        "sigma": 1,
        "noise": 0.15,
        "type": "moons",
        "noise_type": 0,
        "num": 128,
        "kmu": 5,
        "ckmu": 20,
        "factor": 0.3,
    },
    "circles": {
        "norm": 0,
        "gmu": 1,
        "sigma": 1,
        "noise": [float(i / 20) for i in range(6)],
        "type": "circles",
        "noise_type": 0,
        "num": 128,
        "kmu": 5,
        "ckmu": 20,
        "factor": 0.3,
    },
}


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
    dd = DealData("demo", param)
    dd.deal_demo()
    dd.get_demo()
    # dd = DealData("synthesis", param)
    # dd.deal_synthesis()
    # dd.get_synthesis()
    # dd = DealData("uci", param)
    # dd.deal_uci()
    # dd.get_uci()


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
    # test_dataset("synthesis")
    # test_dataset("uci")


def generate_demo_data():
    """
    生成 demo 数据集
    """
    """moons 数据集"""
    param = {
        "norm": 0,
        "gmu": 1,
        "sigma": 1,
        "noise": 0.15,
        "type": "moons",
        "noise_type": 0,
        "num": 128,
        "mu": 10,
        "factor": 0.3,
    }
    dd = DealData("demo", param)
    dd.deal_demo()
    dd.get_demo()
    """circles 数据集"""
    param = {
        "norm": 0,
        "gmu": 1,
        "sigma": 1,
        "noise": 0.15,
        "type": "circles",
        "noise_type": 0,
        "num": 128,
        "mu": 10,
        "factor": 0.3,
    }
    dd = DealData("demo", param)
    for i in range(6):
        dd.params["noise"] = float(i / 20)
        dd.deal_demo()
        dd.get_demo()


def test_run_demo():
    """ """

    rd = RunDemo("./dataset/experiment/demo/", params)
    rd.deal_moons()
    rd.deal_circles()


def fun(i):
    print("Fun: " + str(i))


class A:
    def __init__(self, i) -> None:
        self.i = i

    def cluster(self):
        print("A: " + str(self.i))


def test_run_synthesis():
    """ """
    rs = RunSynthesis("./dataset/experiment/synthesis/")
    rs.deal_synthesis()


def test_analyze_demo():
    """"""
    ad = AnalyzeDemo("./result/demo/result/", params)
    ad.analyze_demo()


def test_plot_demo():
    """"""
    pd = PlotDemo("./result/demo/analyze/", params)
    # pd.show_moons()
    pd.show_circles()


def test_myplot():
    """"""
    mp = MyPlot("./result/diagram/")
    # mp.improve_rho()
    mp.two_step_assign()


if __name__ == "__main__":
    """"""
    # test_process()
    # test_compare()
    # test_dpc()
    # test_datasets()
    # generate_demo_data()
    # test_run_demo()
    # print("sss")
    # test_run_synthesis()
    # test_analyze_demo()
    # test_plot_demo()
    test_myplot()
