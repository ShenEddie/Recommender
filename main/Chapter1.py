#!/usr/bin/python
# -*- coding: UTF-8 -*-

# @Time    : 2021-09-22 10:56
# @Author  : Eddie Shen
# @Email   : sheneddie@outlook.com
# @File    : 1.3_system_testing.py
# @Software: PyCharm

# %% Import packages.
import math
from typing import List, Union

# %% Simulate records.
# records[i] = [u, i, rui, pui]
records = [
    [0, 0, 5, 4.8],
    [0, 1, 4, 4.5],
    [1, 0, 3, 5.0],
    [1, 1, 2, 2.3]
]


# %% RMSE.
def rmse(records: List[List[Union[int, float]]]):
    mse = sum([(rui - pui) ** 2 for u, i, rui, pui in records]) / len(records)
    rmse = math.sqrt(mse)
    return rmse


# %% Print result of RMSE.
print(rmse(records))


# %% MAE.
def mae(records: List[List[Union[int, float]]]):
    mae = sum([abs(rui - pui) for u, i, rui, pui in records]) / len(records)
    return mae


# %% Print the result of MAE.
print(mae(records))
