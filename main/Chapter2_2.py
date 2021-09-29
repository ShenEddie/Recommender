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
def get_items_pool(data: List[List[int]]) -> List[int]:
    return [line[1] for line in data].copy()


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


# data = load_m1_1m()
data = [('a', 101, 1), ('a', 111, 1), ('a', 141, 0),
        ('b', 111, 0), ('b', 151, 1), ('b', 131, 0),
        ('c', 121, 1), ('c', 161, 0), ('c', 141, 0),
        ('d', 111, 1), ('d', 161, 1), ('d', 141, 0), ('d', 121, 0),
        ('e', 131, 1), ('e', 151, 0), ('e', 171, 0),
        ('f', 181, 0), ('f', 191, 1),
        ('g', 101, 1), ('g', 201, 0)]
train_list, test_list = split_data(data, 8, 1)
items_pool = get_items_pool(train_list)
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


# %% Prediction for user-item.
def predict_user_item(user: int,
                      item: int,
                      P: Dict[int, Dict[int, float]],
                      Q: Dict[int, Dict[int, float]],
                      F: int) -> float:
    rui = 0
    for f in range(F):
        rui += P[user][f] * Q[item][f]
    return rui


# %% Latent factor model.
def latent_factor_model(user_items: Dict[int, Dict[int, int]],
                        F: int,
                        n_steps: int,
                        alpha: float,
                        lamb: float,
                        items_pool: List[int]) -> List[Dict[int, Dict[int, float]]]:
    [P, Q] = init_model(user_items, F)
    for step in tqdm(range(0, n_steps)):
        for user, items in user_items.items():
            samples = random_select_negative_sample(items, items_pool)
            for item, rui in samples.items():
                eui = rui - predict_user_item(user, item, P, Q, F)
                for f in range(F):
                    P[user][f] += alpha * (eui * Q[item][f] - lamb * P[user][f])
                    Q[item][f] += alpha * (eui * P[user][f] - lamb * Q[item][f])
        alpha *= 0.9  # Decay learning rate.
    return [P, Q]


P, Q = latent_factor_model(user_items=train_dict,
                           F=100,
                           n_steps=100,
                           alpha=0.02,
                           lamb=0.01,
                           items_pool=items_pool)


# %% Create inverse table for Q.
def inverse_q(Q: Dict[int, Dict[int, float]]
              ) -> Dict[int, Dict[int, float]]:
    Q_inv = dict()
    for i, f_qfi in Q.items():
        for f, qfi in f_qfi.items():
            if Q_inv.get(f):
                Q_inv[f][i] = qfi
            else:
                Q_inv[f] = {i: qfi}
    return Q_inv


Q_inv = inverse_q(Q)


# %% Recommendation for LFM.
def recommend_lfm(user: int,
                  train: Dict[int, Dict[int, int]],
                  P: Dict[int, Dict[int, float]],
                  Q_inv: Dict[int, Dict[int, float]]) -> Dict[int, float]:
    rank = {}
    interacted_item = train[user].keys()
    for f, puf in P[user].items():
        for i, qfi in Q_inv[f].items():
            if i not in interacted_item:
                rank[i] = rank.get(i, 0) + puf * qfi
    return rank


# %% Recall.
def recall_lfm(train: Dict[int, Dict[int, int]],
               test: Dict[int, Dict[int, int]],
               P: Dict[int, Dict[int, float]],
               Q_inv: Dict[int, Dict[int, float]],
               n: int) -> float:
    hit = 0
    all = 0
    for user in tqdm(train.keys()):
        tu = test.get(user, {})
        rank = recommend_lfm(user, train, P, Q_inv)
        rank = sorted(rank.items(), key=itemgetter(1), reverse=True)[0:n]
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += len(tu)
    return hit / all


print(recall_lfm(train_dict, test_dict, P, Q_inv, 10))
