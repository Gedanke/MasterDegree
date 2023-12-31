# -*- coding: utf-8 -*-


from codes import *


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
