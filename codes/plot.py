# -*- coding: utf-8 -*-


import math
import matplotlib.pyplot as plt
import pandas
import numpy
from sklearn.datasets import *
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
import scipy.cluster.hierarchy as sch
from multiprocessing.pool import Pool


"""
绘图
"""
