# -*- coding: utf-8 -*-


from algorithm import *
from .dataSetting import *


def noramlized(data):
    """
    数据归一化处理
    Args:
        data (_type_): 原始数据，不包含标签

    Returns:
        _type_: 归一化后的数据
    """
    """归一化后的数据"""
    new_data = data

    """归一化"""
    new_data = (data - data.min()) / (data.max() - data.min())

    return new_data


def deal_other_synthesis():
    """
    特别处理下其他数据集，如 S2
    """
    raw_data = pandas.read_csv("./dataset/data/synthesis/S2.csv")
    col_list = list(raw_data.columns)
    label = {"label": normalized_label(list(raw_data[col_list[-1]]))}
    data = noramlized(raw_data[col_list[0:-1]]).round(3).join(pandas.DataFrame(label))
    data.columns = list(range(len(data.columns) - 1)) + ["label"]
    data.to_csv("./dataset/data/synthesis/S2.csv", index=False)


def deal_other_uci():
    """
    特别处理下其他数据集，如 txt，data 格式，主要是将其列标签放到最后一列，其他的不动
    """
    file = "abalone"
    dir_path = "./dataset/raw/uci/" + file + "/" + file + ".csv"
    raw_data = pandas.read_csv(dir_path)
    col_list = list(raw_data.columns)
    label = {"label": normalized_label(list(raw_data[col_list[-1]]))}
    data = noramlized(raw_data[col_list[0:-1]]).round(3).join(pandas.DataFrame(label))
    data.columns = list(range(len(data.columns) - 1)) + ["label"]
    print(data)
    # data.to_csv(dir_path, index=False)


def test_normalized_label():
    """ """
    label = [1, 2, 2, 1, 1, 2, 3, 2, 1, 2, 3, 2]
    print(normalized_label(label))
    print(label)


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


import pandas
from multiprocessing.pool import *


# class A:
#     """"""

#     def __init__(self) -> None:
#         """"""

#     def m(self, id):
#         """

#         Args:
#             id (_type_): _description_
#         """
#         print(id)
#         pandas.DataFrame({"A": [1, 2, 3, 0, id]}).to_csv(str(id) + ".csv")

#     def a(self):
#         """ """
#         p = Pool()
#         p.apply_async(self.m, args=(1))
#         p.apply_async(self.m, args=(2))
#         p.apply_async(self.m, args=(3))
#         p.apply_async(self.m, args=(4))
#         p.close()
#         p.join()

# s_num=10
# dis_m=[
#     0 for _ in range(int(s_num*(s_num-1)/2))
# ]
# dis_mm=[
#     0 for _ in range(int(s_num*(s_num-1)/2))
# ]
# num=0
# for i in range(s_num):
#     for j in range(i+1,s_num):
#         dis_m[num]=i+j
#         num+=1

# dis_mm=[
#     i+j for i in range(s_num) for j in range(i+1,s_num)
# ]
# print(dis_m)
# print(dis_mm)
if __name__ == "__main__":
    """"""
    # deal_other_uci()
    # file = "abalone"
    # dir_path = "./dataset/raw/uci/" + file + "/" + file + ".csv"
    # raw_data = pandas.read_csv("tmp.csv")
    # col_list = list(raw_data.columns)
    # label = {"label": normalized_label(list(raw_data[col_list[-1]]))}
    # data = noramlized(raw_data[col_list[0:-1]]).round(3).join(pandas.DataFrame(label))
    # data.columns = list(range(len(data.columns) - 1)) + ["label"]
    # print(data)
