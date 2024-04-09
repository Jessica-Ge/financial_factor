# -*- coding: utf-8 -*-
"""
Created at 2022/11/24 10:19

@author: SY
"""
import pandas as pd
import numpy as np


# -------------------------------------------------------------------------
# 中位数去极值
# factor          因子矩阵, 行为日期, 列为股票代码
# -------------------------------------------------------------------------
def winsorize(factor):

    # 计算上下限：中位数+-5*MAD
    med = np.nanmedian(factor, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(factor - med), axis=1, keepdims=True)
    ub = med + 5 * mad
    lb = med - 5 * mad
    # 超过上下限则设为上下限
    _factor = factor.values
    for dt in range(_factor.shape[0]):
        if mad[dt] != 0:
            _factor[dt, _factor[dt, :] > ub[dt]] = ub[dt]
            _factor[dt, _factor[dt, :] < lb[dt]] = lb[dt]
    factor.loc[:, :] = _factor

    return factor


# -------------------------------------------------------------------------
# 因子标准化
# factor          因子矩阵, 行为日期, 列为股票代码
# -------------------------------------------------------------------------
def normalize(factor):

    _factor = factor.sub(factor.mean(axis=1), axis=0).div(factor.std(axis=1), axis=0)
    # 若标准差为nan, 则保留原数据
    _factor.loc[factor.std(axis=1).isnull(), :] = factor.loc[factor.std(axis=1).isnull(), :]

    return _factor
