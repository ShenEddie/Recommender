#!/usr/bin/python
# -*- coding: UTF-8 -*-

# @Time    : 2021-09-24 13:57
# @Author  : Eddie Shen
# @Email   : sheneddie@outlook.com
# @File    : UserCF.py
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


# %% User similarity.
def user_similarity(
        train: Dict[int, Dict[int, int]]
) -> Dict[int, Dict[int, float]]:
    # Build inverse table for item_users.
    item_users = dict()
    for user, items in tqdm(train.items()):
        for item in items.keys():
            if item_users.get(item):
                item_users[item].add(user)
            else:
                item_users[item] = {user}

    # Calculate co-rated items between users.
    C = dict()
    N = dict()
    for i, users in tqdm(item_users.items()):
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
    for u, related_users in tqdm(C.items()):
        for v, cuv in related_users.items():
            if W.get(u):
                W[u][v] = cuv / math.sqrt(N[u] * N[v])
            else:
                W[u] = {v: cuv / math.sqrt(N[u] * N[v])}
    return W


# %% User similarity iif.
def user_similarity_iif(train):
    # Build inverse table for item_users.
    item_users = dict()
    for user, items in tqdm(train.items()):
        for item in items.keys():
            if item_users.get(item):
                item_users[item].add(user)
            else:
                item_users[item] = {user}

    # Calculate co-rated items between users.
    C = dict()
    N = dict()
    for i, users in tqdm(item_users.items()):
        for u in users:
            N[u] = N.get(u, 0) + 1
            for v in users:
                if u == v:
                    continue
                else:
                    if C.get(u):
                        C[u][v] = C[u].get(v, 0) + 1 / math.log(1 + len(users))
                    else:
                        C[u] = {v: 1 / math.log(1 + len(users))}

    # Calculate finial similarity matrix W.
    W = dict()
    for u, related_users in tqdm(C.items()):
        for v, cuv in related_users.items():
            if W.get(u):
                W[u][v] = cuv / math.sqrt(N[u] * N[v])
            else:
                W[u] = {v: cuv / math.sqrt(N[u] * N[v])}
    return W


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


# %% Recall.
def recall_user_cf(train: Dict[int, Dict[int, int]],
                   test: Dict[int, Dict[int, int]],
                   W: Dict[int, Dict[int, float]],
                   k: int,
                   n: int) -> float:
    hit = 0
    all = 0
    for user in tqdm(train.keys()):
        tu = test.get(user, {})
        rank = recommend_user_cf(user, train, W, k)
        rank = sorted(rank.items(), key=itemgetter(1), reverse=True)[0:n]
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += len(tu)
    return hit / all


# %% Precision
def precision_user_cf(train: Dict[int, Dict[int, int]],
                      test: Dict[int, Dict[int, int]],
                      W: Dict[int, Dict[int, float]],
                      k: int,
                      n: int) -> float:
    hit = 0
    all = 0
    for user in tqdm(train.keys()):
        tu = test.get(user, {})
        rank = recommend_user_cf(user, train, W, k)
        rank = sorted(rank.items(), key=itemgetter(1), reverse=True)[0:n]
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += len(rank)
    return hit / all


# %% Coverage.
def coverage_user_cf(train: Dict[int, Dict[int, int]],
                     test: Dict[int, Dict[int, int]],
                     W: Dict[int, Dict[int, float]],
                     k: int,
                     n: int) -> float:
    recommend_items = set()
    all_items = set()
    for user in tqdm(train.keys()):
        for item in train[user].keys():
            all_items.add(item)
        rank = recommend_user_cf(user, train, W, k)
        rank = sorted(rank.items(), key=itemgetter(1), reverse=True)[0:n]
        for item, pui in rank:
            recommend_items.add(item)
    coverage_rate = len(recommend_items) / len(all_items)
    return coverage_rate


# %% Popularity.
def popularity_user_cf(train: Dict[int, Dict[int, int]],
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
        rank = recommend_user_cf(user, train, W, k)
        rank = sorted(rank.items(), key=itemgetter(1), reverse=True)[0:n]
        for item, pui in rank:
            ret += math.log(1 + item_popularity[item])
            num += 1
    ret /= num
    return ret


if __name__ == '__main__':
    data = load_m1_1m()
    train_list, test_list = split_data(data, 8, 1)
    train_dict = transfer_list2dict(train_list)
    test_dict = transfer_list2dict(test_list)

    del train_list, test_list, data
    gc.collect()

    # train_sample = {
    #     1: {1: 1, 2: 1, 4: 1},
    #     2: {1: 1, 3: 1},
    #     3: {2: 1, 5: 1},
    #     4: {3: 1, 4: 1, 5: 1}
    # }
    #
    # W_sample = user_similarity(train_sample)
    W = user_similarity(train_dict)
    W_iif = user_similarity_iif(train_dict)
    # rank_sample = recommend_user_cf(1, train_sample, W_sample, 3)

    print(recall_user_cf(train_dict, test_dict, W, 80, 10))
    print(recall_user_cf(train_dict, test_dict, W_iif, 80, 10))
    print(precision_user_cf(train_dict, test_dict, W, 80, 10))
    print(precision_user_cf(train_dict, test_dict, W_iif, 80, 10))
    print(coverage_user_cf(train_dict, test_dict, W, 80, 10))
    print(coverage_user_cf(train_dict, test_dict, W_iif, 80, 10))
    print(popularity_user_cf(train_dict, test_dict, W, 80, 10))
    print(popularity_user_cf(train_dict, test_dict, W_iif, 80, 10))
