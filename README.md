# readme

学位论文的相关代码

---

## 项目结构

* [codes](codes) 代码块
  * [algorithm](./codes/algorithm/) 算法模块
    * [__init__.py](./codes/algorithm/__init__.py) 包文件
    * [setting.py](./codes/algorithm/setting.py) 一些公用的函数，以及数据集的基本信息
    * [compare.py](./codes/algorithm/compare.py) 对 sklearn 中的算法进行封装
    * [dpc.py](./codes/algorithm/dpc.py) dpc 基类算法
    * [dpcs.py](./codes/algorithm/dpcs.py) dpc 系列对比算法
    * [dpcm.py](./codes/algorithm/dpcm.py) 本文改进的三个 DPC 算法，以及将三个改进变为可选选项融入到 DPC 中
  * [__init__.py](./codes/__init__.py) 包文件
  * [process.py](./codes/process.py) 数据处理文件
  * [dataSetting.py](./codes/dataSetting.py) 一些公用的函数，不同算法在不同数据集上的设置参数
  * [experiment.py](./codes/experiment.py) 算法根据设置的参数在不同数据运行获得实验结果
  * [analyze.py](./codes/analyze.py) 对算法在数据集上获得的结果进行分析
  * [plot.py](./codes/plot.py) 根据结果绘制图
  * [conf.json](./codes/conf.json) 不同算法在不同数据集上的结果记录
* [dataset](dataset) 数据块
  * [raw](dataset/raw) 原始数据
    * [demo](dataset/raw/demo/)
    * [synthesis](dataset/raw/synthesis/)：人工合成数据集，以二维为主，结构相对简单，易于作图
    * [uci](dataset/raw/uci)
    * [image](dataset/raw/image/)
  * [data](dataset/data) 预处理完成后的数据
    * [demo](dataset/data/demo/)
    * [synthesis](dataset/data/synthesis/)
    * [uci](dataset/data/uci)
    * [image](dataset/data/image/)
  * [experiment](dataset/experiment/) 实验需要用的数据，从 [data](dataset/data/) 中提取，加入了相关参数；如果不加，直接复制过来
    * [demo](dataset/experiment/demo/)
    * [synthesis](dataset/experiment/synthesis/)
    * [uci](dataset/experiment/uci)
    * [image](dataset/experiment/image/)
* [result](result) 结果块，均以 pandas.DataFrame 形式存放，以追加的形式增加到现有数据中
  * [demo](result/demo) 结构同下
  * [mnist](result/mnist) 结构同下
  * [uci](result/uci) 存放结果(多进程下，每一个类创建属于自己的结果文件)
    * [plot](result/demo/plot) 存放聚类结果图
    * [result](result/demo/result) 存放聚类结果数据，只存放预测的标签列(列名为此次算法运行的参数，列内容为标签，顺序与原始数据索引一致)，存放聚类结果指标
    * [analyze](result/demo/analyze) 存放分析结果(列名为参数+聚类指标，行为每一次不同参数下的实验结果)
* [README](README.md) readme

注意下：项目中所有涉及到参数的索引，均从0开始，不以1开始

---

## 数据集

### 合成数据集

|   Dataset   | Number of instances | Number of features | Number of classes | Source |
| :---------: | :-----------------: | :----------------: | :---------------: | :----: |
| Aggregation |         788         |         2         |         7         |        |
|  Compound  |         399         |         2         |         2         |        |
|     D31     |        3100        |         2         |        31        |        |
|   DIM512   |        1024        |        512        |        16        |        |
|    Flame    |         240         |         2         |         2         |        |
|    Jain    |         373         |         2         |         2         |        |
|  Pathbased  |         300         |         2         |         3         |        |
|     R15     |         600         |         2         |        15        |        |
|     S2     |        5000        |         2         |        15        |        |
|   Spiral   |         412         |         2         |         3         |        |

Digits
USPS
MNIST-test
Fashion
Letter

https://github.com/gagolews/clustering-data-v0

https://clustering-benchmarks.gagolewski.com/


## UCI 数据集

该部分的 UCI 数据集的信息如下，如果一个数据集的名称为 dataset

* dataset.data 原始未经处理的数据集，标签统一放在最后一列，分隔符统一为 ,(排除仅含 dataset.mat 外，原始数据集一定存在)
* dataset.names 对该数据集的详细介绍(大体上与 data 一同出现)
* dataset.txt 对该数据集的简略介绍(大体上与 data 一同出现)
* dataset.mat mat 格式的数据，可以直接读取，但不保证 key 是相同(一定存在该数据集)
* minmax_dataset.mat 经过归一化后的数据集(大部分存在)

Real-world datasets

|    Dataset    | Number of instances | Number of  features | Number of classes | Source |
| :-----------: | :-----------------: | :-----------------: | :---------------: | :----: |
|    Abalone    |        4177        |          8          |         3         |        |
|     Blood     |         784         |          5          |         2         |        |
|    Coil20    |                    |                    |                  |        |
|  Dermatology  |         366         |         34         |         6         |        |
|     Ecoil     |         336         |          7          |         8         |        |
|     Glass     |         214         |          9          |         6         |        |
|     Iris     |         150         |          4          |         3         |        |
|    Isolet    |        1560        |         617         |        26        |        |
|     Jaffe     |         213         |        65536        |        10        |        |
|    Letter    |        20000        |         16         |        26        |        |
|    libras    |         360         |         90         |        15        |        |
|     Lung     |         203         |        3312        |         5         |        |
|     Magic     |        19020        |         10         |         2         |        |
|  Parkinsons  |         195         |         23         |         2         |        |
|     Pima     |         768         |          8          |         2         |        |
|     Seeds     |         210         |          7          |         3         |        |
|    Segment    |        2310        |         19         |         7         |        |
|     Sonar     |         208         |         60         |         2         |        |
|   Spambase   |        4601        |         57         |         2         |        |
|   Teaching   |         151         |          5          |         3         |        |
|    Tox_171    |         171         |        5748        |         4         |        |
|    Twonorm    |        7400        |         20         |         2         |        |
|     Usps     |        11000        |         256         |        10        |        |
|   Waveform   |        5000        |         21         |         3         |        |
| WaveformNoise |        5000        |         40         |         3         |        |
|     Wdbc     |         569         |         30         |         2         |        |
|     Wilt     |        4839        |         13         |         3         |        |
|     Wine     |         178         |         13         |         3         |        |

---
