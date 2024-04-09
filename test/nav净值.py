# -*- coding: utf-8 -*-
# Author: Jessica
# Creat Time: 2024/3/31
# FileName: nav净值
# Description: simple introduction of the code

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import sys
import os
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap

from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap, centered_cmap
from plottable.formatters import decimal_to_percent
from plottable.plots import circled_image

# sns.set_theme(style='darkgrid')  # 图形主题
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams["savefig.bbox"] = "tight"  # 图形保存时去除白边

from utils import BacktestUtils, BacktestUtilsOpenClose, PerfUtils, FactorProcess

import warnings
warnings.filterwarnings("ignore")


# -------------------------------------------------------------------------
# 分层回测
# [输入]
# factor
# start_date     计算回测净值曲线开始时间
# end_date       计算回测净值曲线终止时间
# base           参照基准, 指数代码
# lay_number     分层数, 默认5层
# -------------------------------------------------------------------------

if __name__ == '__main__':

    stock_close = pd.read_pickle(r'D:\Desktop\营业利润财务质量\data\stock_close_backtest')
    stock_open = pd.read_pickle(r'D:\Desktop\营业利润财务质量\data\stock_close_backtest')

    # 个股收盘价
    stock_close.index = pd.to_datetime(stock_close.index, format="%Y%m%d")
    # 个股开盘价
    stock_open.index = pd.to_datetime(stock_open.index, format="%Y%m%d")

    # 日频交易日期
    daily_dates = stock_open.index.tolist()
    # 所有日频交易日期序列
    date_series = pd.Series(daily_dates, index=daily_dates)

    # =============================================================================
    # 因子评估
    # =============================================================================

    print('因子评价-0409')

    factor_name = ['基于营业利润计算过程的财务质量因子']
    factors_dict = pd.read_pickle(r'D:\Desktop\营业利润财务质量\data\backtest_factor_2')
    #factors_dict = abs(factors_dict)

    # 将季度factor处理成月度factor
    # factors_month = factors_dict.resample('M').last().copy()
    # factors_month = factors_month.fillna(method='bfill', axis=0)
    factors_month = factors_dict.loc['2016-04-30':, :]

    # eval_model.factor_eval
    factor_orig = factors_month.dropna(how='all', axis=1)
    factor_orig = factor_orig.dropna(how='all', axis=0)
    factor_orig = factor_orig.replace([np.inf, -np.inf], np.nan)

    # 因子预处理
    print('因子预处理: Winsorize')
    factor_processed = FactorProcess.winsorize(factor_orig)
    print('因子预处理: Normalize')
    factor_processed = FactorProcess.normalize(factor_processed)

    #分层回测
    # eval_model.factor_eval.clsfy_backtest
    origin_factor = factor_processed
    freq = 'M'
    start_date = factor_processed.index[0]
    end_date = factor_processed.index[-1]
    layer_number = 10

    # 回测净值
    nav = pd.DataFrame(columns=['分层1', '分层2', '分层3', '分层4', '分层5',
                                '分层6', '分层7', '分层8', '分层9', '分层10', '基准'])

    merge_factor = factor_processed.resample(freq).last().copy()
    merge_factor = merge_factor.dropna(how='all')

    # 当指标给定的最后一个日期大于最后一个交易日时（月末容易出现）
    # 最后一个交易信号无法调仓
    if merge_factor.index[-1] >= daily_dates[-1]:
        merge_factor = merge_factor.drop(index=merge_factor.index[-1])

    # 调仓日期为生成信号的下一天，即月度初的第一个交易日
    daily_dates = pd.Series(data=daily_dates, index=daily_dates)
    merge_factor.index = [daily_dates[daily_dates > i].index[0] for i in merge_factor.index]

    # 指标进行排序
    merge_factor_rank = merge_factor.rank(method='average', ascending=False, axis=1)

    # 各层持仓权重
    port = []

    # 回测,计算策略净值
    # 划分各层归属
    for layer_id in tqdm(range(layer_number), desc='分层回测中', leave=False):
        # 方案2: 仅从存在因子值的股票中进行分层, 每层股票数在不同调仓期可能会改变
        thres_up = 1 / layer_number * (layer_id + 1)  #该层占所有股票的权重的上限
        thres_down = 1 / layer_number * layer_id     #该层占所有股票的权重的下限

        factor_layer = pd.DataFrame(np.zeros_like(merge_factor.values), index=merge_factor.index,
                                    columns=merge_factor.columns)

        # 选出哪些股票属于这一层，左开右闭
        factor_layer[(merge_factor_rank.apply(lambda x: x > x.max() * thres_down, axis=1)) &
                     (merge_factor_rank.apply(lambda x: x <= x.max() * thres_up, axis=1))] = 1

        # 全为零行替换为全仓
        factor_layer[(factor_layer == 0).sum(axis=1) == factor_layer.shape[1]] = 1 / factor_layer.shape[1]

        # 空值替换为全仓
        factor_layer[factor_layer.isnull().sum(axis=1) == factor_layer.shape[1]] = 1 / factor_layer.shape[1]

        # 无因子值的股票权重置为0
        factor_layer[merge_factor_rank.isnull()] = 0

        # 持仓归一化
        factor_layer = (factor_layer.multiply(1 / factor_layer.sum(axis=1), axis=0)).loc[start_date:end_date, :]

        port.append(factor_layer)

        # 回测,计算策略净值f
        nav[f'分层{layer_id + 1}'], _ = BacktestUtilsOpenClose.cal_nav(factor_layer,
                                                                     stock_open.loc[start_date:end_date,
                                                                     stock_open.columns.intersection(factor_layer.columns)],
                                                                     stock_close.loc[start_date:end_date,
                                                                     stock_close.columns.intersection(factor_layer.columns)],
                                                                     base_nav=0,
                                                                     fee=0)  # 有6个股票没有开售盘价







