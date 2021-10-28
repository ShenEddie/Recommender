#!/usr/bin/python
# -*- coding: UTF-8 -*-

# @Time    : 2021-09-27 09:29
# @Author  : Eddie Shen
# @Email   : sheneddie@outlook.com
# @File    : ItemCF.py
# @Software: PyCharm


# %% Import packages.
import os
import sys
import gc
import math
import random
import copy
import pickle
from typing import List, Dict, Union
from operator import itemgetter
from tqdm import tqdm
from pprint import pprint

sys.path.append('../')
try:
    from utils.load_data import load_m1_1m, transfer_list2dict
    from config.file_path import item_cf_W_path, item_cf_W_iuf_path
    from config.file_path import item_cf_W_penalty_path, item_cf_W_normed_path
except (ModuleNotFoundError, ImportError):
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


# %% Item similarity iuf.
def item_similarity_iuf(
        train: Dict[int, Dict[int, int]]
) -> Dict[int, Dict[int, float]]:
    # Calculate co-rated users between items.
    C: Dict[int, Dict[int, float]] = dict()
    N: Dict[int, int] = dict()
    for u, items in tqdm(train.items()):
        for i in items.keys():
            N[i] = N.get(i, 0) + 1
            for j in items.keys():
                if i == j:
                    continue
                else:
                    if C.get(i):
                        C[i][j] = C[i].get(j, 0) + 1 / math.log(1 + len(items))
                    else:
                        C[i] = {j: 1 / math.log(1 + len(items))}
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


# %% Item Similarity: increase the penalty for hot items.
def item_similarity_penalty(
        train: Dict[int, Dict[int, int]],
        alpha: float
) -> Dict[int, Dict[int, Union[int, float]]]:
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

    # Calculate finial similarity matrix W.
    W: Dict[int, Dict[int, float]] = dict()
    for i, related_items in tqdm(C.items()):
        for j, cij in related_items.items():
            if W.get(i):
                W[i][j] = cij / ((N[i] ** (1 - alpha)) * (N[j] ** alpha))
            else:
                W[i] = {j: cij / ((N[i] ** (1 - alpha)) * (N[j] ** alpha))}
    return W


# %% Normalize similarity matrix.
def similarity_norm(W: Dict[int, Dict[int, Union[int, float]]]
                    ) -> Dict[int, Dict[int, float]]:
    W_normed = copy.deepcopy(W)  # !!!
    for i, wi in tqdm(W.items()):
        max_wij = max(wi.values())
        for j, wij in wi.items():
            W_normed[i][j] = wij / max_wij
    return W_normed


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


if __name__ == '__main__':
    data = load_m1_1m()
    train_list, test_list = split_data(data, 8, 1)
    train_dict = transfer_list2dict(train_list)
    test_dict = transfer_list2dict(test_list)

    del train_list, test_list, data
    gc.collect()

    # Calculate weight matrix W.
    if os.path.isfile(item_cf_W_path):
        W = pickle.load(open(item_cf_W_path, 'rb'))
    else:
        W = item_similarity(train_dict)
        pickle.dump(W, open(item_cf_W_path, 'wb'))

    # Test ItemCF.
    res = {
        'K': [],
        'recall': [],
        'precision': [],
        'coverage': [],
        'popularity': []
    }
    N = 10  # Top-N recommendation.
    for K in [5, 10, 20, 40, 80, 100]:
        res['K'].append(K)
        recall = recall_item_cf(train_dict, test_dict, W, K, N)
        res['recall'].append(recall)
        precision = precision_item_cf(train_dict, test_dict, W, K, N)
        res['precision'].append(precision)
        coverage = coverage_item_cf(train_dict, test_dict, W, K, N)
        res['coverage'].append(coverage)
        popularity = popularity_item_cf(train_dict, test_dict, W, K, N)
        res['popularity'].append(popularity)
        print("K:{}, recall:{}, precision:{}, coverage:{}, popularity:{}"
              "".format(K, recall, precision, coverage, popularity))
    print(res)

    del W
    gc.collect()

    # Calculate weight matrix W_iuf.
    if os.path.isfile(item_cf_W_iuf_path):
        W_iuf = pickle.load(open(item_cf_W_iuf_path, 'rb'))
    else:
        W_iuf = item_similarity_iuf(train_dict)
        pickle.dump(W_iuf, open(item_cf_W_iuf_path, 'wb'))

    # Test ItemIUF.
    K = 10
    recall = recall_item_cf(train_dict, test_dict, W_iuf, K, N)
    precision = precision_item_cf(train_dict, test_dict, W_iuf, K, N)
    coverage = coverage_item_cf(train_dict, test_dict, W_iuf, K, N)
    popularity = popularity_item_cf(train_dict, test_dict, W_iuf, K, N)
    print("User_iif: recall:{}, precision:{}, coverage:{}, popularity:{}"
          "".format(recall, precision, coverage, popularity))
    res_iuf = [recall, precision, coverage, popularity]
    print(res_iuf)

    del W_iuf
    gc.collect()

    # Calculate weight matrix W_normed.
    if os.path.isfile(item_cf_W_normed_path):
        W_normed = pickle.load(open(item_cf_W_normed_path, 'rb'))
    else:
        W = pickle.load(open(item_cf_W_path, 'rb'))
        W_normed = similarity_norm(W)
        pickle.dump(W_normed, open(item_cf_W_normed_path, 'wb'))

    # Test ItemCF_Normed.
    recall = recall_item_cf(train_dict, test_dict, W_normed, K, N)
    precision = precision_item_cf(train_dict, test_dict, W_normed, K, N)
    coverage = coverage_item_cf(train_dict, test_dict, W_normed, K, N)
    popularity = popularity_item_cf(train_dict, test_dict, W_normed, K, N)
    print("User_iif: recall:{}, precision:{}, coverage:{}, popularity:{}"
          "".format(recall, precision, coverage, popularity))
    res_normed = [recall, precision, coverage, popularity]
    print(res_normed)

    del W_normed
    gc.collect()

    # Calculate weight matrix W_penalty.
    if os.path.isfile(item_cf_W_penalty_path):
        W_penalty = pickle.load(open(item_cf_W_penalty_path, 'rb'))
    else:
        W_penalty = item_similarity_penalty(train_dict, 0.55)
        pickle.dump(W_penalty, open(item_cf_W_penalty_path, 'wb'))

    # Test ItemCF_Penalty.
    res_penalty = {
        'alpha': [],
        'recall': [],
        'precision': [],
        'coverage': [],
        'popularity': []
    }
    N = 10  # Top-N recommendation.
    for alpha in [0.4, 0.5, 0.55, 0.6, 0.7]:
        W_penalty = item_similarity_penalty(train_dict, alpha)
        res_penalty['alpha'].append(alpha)
        recall = recall_item_cf(train_dict, test_dict, W_penalty, K, N)
        res_penalty['recall'].append(recall)
        precision = precision_item_cf(train_dict, test_dict, W_penalty, K, N)
        res_penalty['precision'].append(precision)
        coverage = coverage_item_cf(train_dict, test_dict, W_penalty, K, N)
        res_penalty['coverage'].append(coverage)
        popularity = popularity_item_cf(train_dict, test_dict, W_penalty, K, N)
        res_penalty['popularity'].append(popularity)
        print("alpha:{}, recall:{}, precision:{}, coverage:{}, popularity:{}"
              "".format(alpha, recall, precision, coverage, popularity))
    print(res_penalty)
