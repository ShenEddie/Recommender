#!/usr/bin/python
# -*- coding: UTF-8 -*-

# @Time    : 2021-09-24 13:57
# @Author  : Eddie Shen
# @Email   : sheneddie@outlook.com
# @File    : Chapter2.py
# @Software: PyCharm

# %% Import packages.
import sys
import math
import random
from typing import List, Dict
from operator import itemgetter

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


# %% User similarity.
def user_similarity(train: Dict[int, Dict[int, int]]) -> Dict[int, Dict[int, float]]:
    # Build inverse table for item_users.
    item_users = dict()
    for user, items in train.items():
        for item in items.keys():
            if item_users.get(item):
                item_users[item].add(user)
            else:
                item_users[item] = {user}

    # Calculate co-rated items between users.
    C = dict()
    N = dict()
    for i, users in item_users.items():
        for u in users:
            N[u] = N.get(u, 0) + 1
            for v in users:
                if u == v:
                    continue
                else:
                    if C.get(u):
                        C[u][v] = C[u].get(v, 0) + 1
                    else:
                        C[u] = {v: 1}

    # Calculate finial similarity matrix W.
    W = dict()
    for u, related_users in C.items():
        for v, cuv in related_users.items():
            if W.get(u):
                W[u][v] = cuv / math.sqrt(N[u] * N[v])
            else:
                W[u] = {v: cuv / math.sqrt(N[u] * N[v])}
    return W


train_sample = {
    1: {1: 1, 2: 1, 4: 1},
    2: {1: 1, 3: 1},
    3: {2: 1, 5: 1},
    4: {3: 1, 4: 1, 5: 1}
}

train_dict = transfer_list2dict(train_list)
W_sample = user_similarity(train_sample)
W = user_similarity(train_dict)


# %% UserCF algorithm.
def recommend_user_cf(user: int,
                      train: Dict[int, Dict[int, int]],
                      W: Dict[int, Dict[int, float]],
                      K: int):
    rank = dict()
    interacted_item = train[user]
    for v, wuv in sorted(W[user].items(), key=itemgetter(1), reverse=True)[0:K]:
        for i, rvi in train[v].items():
            if i not in interacted_item:
                rank[i] = rank.get(i, 0) + wuv * rvi
    return rank


rank_sample = recommend_user_cf(1, train_sample, W_sample, 3)