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


# %% Init matrix P & Q.
def init_model(user_items: Dict[int, Dict[int, int]],
               F: int) -> List[Dict[int, Dict[int, float]]]:
    P = dict()
    Q = dict()
    all_items = set()
    for u, items in user_items.items():
        for item in items.keys():
            all_items.add(item)
        for f in range(F):
            if P.get(u):
                P[u][f] = random.random() / math.sqrt(F)
            else:
                P[u] = {f: random.random() / math.sqrt(F)}

    for i in all_items:
        for f in range(F):
            if Q.get(i):
                Q[i][f] = random.random() / math.sqrt(F)
            else:
                Q[i] = {f: random.random() / math.sqrt(F)}
    return [P, Q]


