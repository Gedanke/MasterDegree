# -*- coding: utf-8 -*-


import os
import sys

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
        "num": 1024,
        "kmu": 2,
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
        "num": 2048,
        "kmu": 2,
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
        "num": 1024,
        "mu": 4,
        "factor": 0.5,
    }
    # dd = DealData("demo", param)
    # dd.deal_demo()
    # dd.get_demo()
    # dd = DealData("synthesis", param)
    # dd.deal_synthesis()
    # dd.get_synthesis()
    # dd = DealData("uci", param)
    # dd.deal_uci()
    # dd.get_uci()
    dd = DealData("image", param)
    dd.deal_image()


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
        "num": 1024,
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
        "num": 2048,
        "mu": 10,
        "factor": 0.3,
    }
    dd = DealData("demo", param)
    for i in range(6):
        dd.params["noise"] = float(i / 20)
        dd.deal_demo()
        dd.get_demo()


def test_run_demo():
    """
    运行 demo 数据集
    """
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
    # rs.deal_synthesis()


def test_run_uci():
    """"""
    dl = ["iris"]
    al = ["DpcD"]
    ru = RunUci("./dataset/experiment/uci/", dl, al)
    ru.deal_uci()


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
    mp = MyPlot("./result/", "./result/diagram/")
    """两步样本分配策略演示图，第一张图"""
    mp.two_step_assign1()
    """两步样本分配策略演示图，第二张图"""
    mp.two_step_assign2()
    """两步样本分配策略演示图，第三张图"""
    mp.two_step_assign3()
    """两步样本分配策略演示图，第四张图"""
    mp.two_step_assign4()
    # import matplotlib.font_manager as fm

    # l = [font.name for font in fm.fontManager.ttflist]
    # print("SimSun" in l)


def test_analyze_synthesis():
    """"""
    as_ = AnalyzeSynthesis()
    as_.analyze_synthesis()


def test_analyze_uci():
    """"""
    as_ = AnalyzeUci()
    as_.analyze_uci()


def test_plot_synthesis():
    """"""
    ps = PlotSynthesis()
    # ps.show_dpc_process()
    # ps.show_dpc_compare()
    # ps.show_rho_compare()
    # ps.show_distance_compare()
    # ps.show_percent_compare()
    # ps.show_cluster_results()
    ps.show_lw_dpc_defect()


def test_distance_dpc():
    """ """
    data_name = "flame"
    path = SYNTHESIS_PARAMS[data_name]["path"]
    save_path = SYNTHESIS_PARAMS[data_name]["save_path"]
    num = SYNTHESIS_PARAMS[data_name]["num"]
    _percent = 2.8
    distance_method = "euclidean"
    DpcD(path, save_path, num, _percent, 1, 0, 0, distance_method)
    distance_method = "cityblock"
    DpcD(path, save_path, num, _percent, 1, 0, 0, distance_method)
    distance_method = "chebyshev"
    DpcD(path, save_path, num, _percent, 1, 0, 0, distance_method)
    distance_method = "mahalanobis"
    DpcD(path, save_path, num, _percent, 1, 0, 0, distance_method)
    distance_method = "cosine"
    DpcD(path, save_path, num, _percent, 1, 0, 0, distance_method)
    distance_method = "jaccard"
    DpcD(path, save_path, num, _percent, 1, 0, 0, distance_method)


def test_percent_dpc():
    """"""
    data_name = "flame"
    path = SYNTHESIS_PARAMS[data_name]["path"]
    save_path = SYNTHESIS_PARAMS[data_name]["save_path"]
    num = SYNTHESIS_PARAMS[data_name]["num"]
    _percent = 0.5
    Dpc(path, save_path, num, _percent)
    # _percent = 2.0
    # Dpc(path, save_path, num, _percent)
    # _percent = 2.5
    # Dpc(path, save_path, num, _percent)
    # _percent = 3.0
    # Dpc(path, save_path, num, _percent)


def help_show_percent_compare():
    """ """
    data_name = "flame"
    percent_list = [1.0, 2.0, 3.0]

    """3.0 替换下的 label，使其与 1.0，2.0 的绘图颜色一致(只运行一次)"""
    tmp_file = (
        "./result/synthesis/"
        + "result/"
        + data_name
        + "/dpc/"
        + "dcp_"
        + str(percent_list[2])
        + "__dcm_1__rhom_0__dem_0.json"
    )
    with open(
        tmp_file,
        "r",
    ) as f:
        pred_result = json.load(f)
    """替换标签"""
    for idx in range(len(pred_result["label"])):
        if pred_result["label"][idx] == 1:
            pred_result["label"][idx] = 0
        else:
            pred_result["label"][idx] = 1

    """替换中心"""
    pred_result["center"][0], pred_result["center"][1] = (
        pred_result["center"][1],
        pred_result["center"][0],
    )

    """写回文件"""
    with open(tmp_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(pred_result, ensure_ascii=False))


def helper_dpc():
    """ """
    data_name = "D31"
    path = SYNTHESIS_PARAMS[data_name]["path"]
    save_path = SYNTHESIS_PARAMS[data_name]["save_path"]
    num = SYNTHESIS_PARAMS[data_name]["num"]
    # _percent = 1.0
    # Dpc(path, save_path, num, _percent)
    # _percent = 1.2
    # DpcKnn(path, save_path, num, _percent)
    _k = 41
    # SnnDpc(path, save_path, num, _k)
    DpcIRho(
        path,
        save_path,
        num,
        1,
        1,
        0,
        0,
        "euclidean",
        [],
        False,
        {"k": 32, "mu": 10},
    )


def test_plot_uci():
    """ """
    path = "E:/14+/windows10/codes/python/ky/thesisResult/LW_DCP/V6/results/"
    al = ["dpc_Rho_iAss", "dpc_iRho_iAss"]
    """dpc_Rho_iAss"""
    pu = PlotUci(path, al)
    pu.show_euc_k()
    pu.show_ckrod_k()
    pu.show_ckrod_mu()
    pu.show_order_sensitivity()


def test_analyze_image():
    """ """


if __name__ == "__main__":
    """"""
    print("---run---")
    """demo"""
    # generate_demo_data()
    # test_run_demo()
    test_analyze_demo()
    # test_plot_demo()
    # a = 0.1
    # print("{:.2f}".format(a))

    """synthesis"""
    # test_analyze_synthesis()
    # test_plot_synthesis()
    # helper_dpc()
    l = {"a": [0, 1]}
    # print("a" in l.keys())
    # l["a"][0], l["a"][1] = l["a"][1], l["a"][0]
    # print(l)
    # test_distance_dpc()
    # test_percent_dpc()
    # test_process()
    # test_compare()
    # test_dpc()
    # test_datasets()
    # test_run_demo()
    # print("sss")
    # test_run_synthesis()
    # test_myplot()
    # dataset_params = UCI_PATAMS["iris"]
    # path = dataset_params["path"]
    # save_path = dataset_params["save_path"]
    # num = dataset_params["num"]
    # DpcD(path, save_path, num, 0.1, 1, 0, 0, "rod")

    """uci"""
    # test_analyze_synthesis()
    # test_analyze_uci()
    # test_plot_uci()

    """image"""
    test_process()
