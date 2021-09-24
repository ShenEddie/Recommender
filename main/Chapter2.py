#!/usr/bin/python
# -*- coding: UTF-8 -*-

# @Time    : 2021-09-24 13:57
# @Author  : Eddie Shen
# @Email   : sheneddie@outlook.com
# @File    : Chapter2.py
# @Software: PyCharm

# %% Import packages.
import sys
import random
from typing import List

sys.path.append('../')
try:
    from utils.load_data import load_m1_1m
except ModuleNotFoundError:
    raise


# %% Load & Split data.
def split_data(data: List[List[int]], M: int, k: int, seed: int = 1234):
    test = []
    train = []
    random.seed(seed)
    for user, item in data:
        if random.randint(0, M) == k:
            test.append([user, item])
        else:
            train.append([user, item])
    return train, test


data = load_m1_1m()
train_set, test_set = split_data(data, 8, 1)
