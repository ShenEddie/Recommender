#!/usr/bin/python
# -*- coding: UTF-8 -*-

# @Time    : 2021-09-27 09:29
# @Author  : Eddie Shen
# @Email   : sheneddie@outlook.com
# @File    : Chapter2_1.py
# @Software: PyCharm


# %% Import packages.
import sys
import gc
import math
import random
from typing import List, Dict
from operator import itemgetter
from tqdm import tqdm

sys.path.append('../')
try:
    from utils.load_data import load_m1_1m, transfer_list2dict
except ModuleNotFoundError:
    raise


# %% Load & Split data.
def split_data(data: List[List[int]], M: int, k: int, seed: int = 1234):
    test = []
    train = []
    random.seed(seed)
    for user, item, rating in data:
        if random.randint(0, M) == k:
            test.append([user, item, rating])
        else:
            train.append([user, item, rating])
    return train, test


data = load_m1_1m()
train_list, test_list = split_data(data, 8, 1)
train_dict = transfer_list2dict(train_list)
test_dict = transfer_list2dict(test_list)

del train_list, test_list, data
gc.collect()
