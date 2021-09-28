#!/usr/bin/python
# -*- coding: UTF-8 -*-

# @Time    : 2021-09-28 12:06
# @Author  : Eddie Shen
# @Email   : sheneddie@outlook.com
# @File    : Chapter2_2.py
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


# %% Get items' pool
def items_pool(data: List[List[int]]) -> List[int]:
    return [line[1] for line in data].copy()


data = load_m1_1m()
items_pool = items_pool(data)


# %% Select negative sample:
def random_select_negative_sample(items: Dict[int, int],
                                  items_pool: List[int]) -> Dict[int, int]:
    ret = dict()
    for i in items.keys():
        ret[i] = 1  # Positive sample.
    n = 0
    for i in range(0, len(items) * 3):
        item = random.choice(items_pool)
        if item in ret:
            continue
        else:
            ret[item] = 0  # Negative sample.
            n += 1
            if n >= len(items):  # To make sure positive & negative samples are close.
                break
    return ret
