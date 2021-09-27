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
from pprint import pprint

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


# %% Item Similarity.
def item_similarity(train: Dict[int, Dict[int, int]]):
    # Calculate co-rated users between items.
    C: Dict[int, Dict[int, int]] = dict()
    N: Dict[int, int] = dict()
    for u, items in tqdm(train.items()):
        for i in items.keys():
            N[i] = N.get(i, 0) + 1
            for j in items.keys():
                if i == j:
                    continue
                else:
                    if C.get(i):
                        C[i][j] = C[i].get(j, 0) + 1
                    else:
                        C[i] = {j: 1}
    # pprint(C)

    # Calculate finial similarity matrix W.
    W: Dict[int, Dict[int, float]] = dict()
    for i, related_items in tqdm(C.items()):
        for j, cij in related_items.items():
            if W.get(i):
                W[i][j] = cij / math.sqrt(N[i] * N[j])
            else:
                W[i] = {j: cij / math.sqrt(N[i] * N[j])}
    return W


# train_sample = {
#     1: {1: 1, 2: 1, 4: 1},
#     2: {2: 1, 3: 1, 5: 1},
#     3: {3: 1, 4: 1},
#     4: {2: 1, 3: 1, 4: 1},
#     5: {1: 1, 4: 1}
# }
# W_sample = item_similarity(train_sample)
W = item_similarity(train_dict)


# %% ItemCF algorithm.
def recommend_item_cf(user: int,
                      train: Dict[int, Dict[int, int]],
                      W: Dict[int, Dict[int, float]],
                      K: int) -> Dict[int, float]:
    rank = dict()
    ru = train[user]
    for i, pi in ru.items():
        for j, wj in sorted(W[i].items(), key=itemgetter(1), reverse=True)[0:K]:
            if j in ru:
                continue
            else:
                rank[j] = rank.get(j, 0) + pi * wj
    return rank


# %% Recall.
def recall_item_cf(train: Dict[int, Dict[int, int]],
                   test: Dict[int, Dict[int, int]],
                   W: Dict[int, Dict[int, float]],
                   k: int,
                   n: int) -> float:
    hit = 0
    all = 0
    for user in tqdm(train.keys()):
        tu = test.get(user, {})
        rank = recommend_item_cf(user, train, W, k)
        rank = sorted(rank.items(), key=itemgetter(1), reverse=True)[0:n]
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += len(tu)
    return hit / all


print(recall_item_cf(train_dict, test_dict, W, 10, 10))


# %% Precision
def precision_item_cf(train: Dict[int, Dict[int, int]],
                      test: Dict[int, Dict[int, int]],
                      W: Dict[int, Dict[int, float]],
                      k: int,
                      n: int) -> float:
    hit = 0
    all = 0
    for user in tqdm(train.keys()):
        tu = test.get(user, {})
        rank = recommend_item_cf(user, train, W, k)
        rank = sorted(rank.items(), key=itemgetter(1), reverse=True)[0:n]
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += len(rank)
    return hit / all


print(precision_item_cf(train_dict, test_dict, W, 10, 10))


# %% Coverage.
def coverage_item_cf(train: Dict[int, Dict[int, int]],
                     test: Dict[int, Dict[int, int]],
                     W: Dict[int, Dict[int, float]],
                     k: int,
                     n: int) -> float:
    recommend_items = set()
    all_items = set()
    for user in tqdm(train.keys()):
        for item in train[user].keys():
            all_items.add(item)
        rank = recommend_item_cf(user, train, W, k)
        rank = sorted(rank.items(), key=itemgetter(1), reverse=True)[0:n]
        for item, pui in rank:
            recommend_items.add(item)
    coverage_rate = len(recommend_items) / len(all_items)
    return coverage_rate


print(coverage_item_cf(train_dict, test_dict, W, 10, 10))


# %% Popularity.
def popularity_item_cf(train: Dict[int, Dict[int, int]],
                       test: Dict[int, Dict[int, int]],
                       W: Dict[int, Dict[int, float]],
                       k: int,
                       n: int):
    item_popularity = dict()
    for user, items in train.items():
        for item in items.keys():
            item_popularity[item] = item_popularity.get(item, 0) + 1
    ret = 0
    num = 0
    for user in tqdm(train.keys()):
        rank = recommend_item_cf(user, train, W, k)
        rank = sorted(rank.items(), key=itemgetter(1), reverse=True)[0:n]
        for item, pui in rank:
            ret += math.log(1 + item_popularity[item])
            num += 1
    ret /= num
    return ret


print(popularity_item_cf(train_dict, test_dict, W, 10, 10))
