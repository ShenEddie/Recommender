#!/usr/bin/python
# -*- coding: UTF-8 -*-

# @Time    : 2021-09-24 10:59
# @Author  : Eddie Shen
# @Email   : sheneddie@outlook.com
# @File    : load_data.py
# @Software: PyCharm


import sys
from typing import Union, List, Dict

sys.path.append('../')
try:
    from config.file_path import ml_1m_ratings_path
except (ModuleNotFoundError, ImportError):
    raise


def load_m1_1m(path: str = ml_1m_ratings_path,
               if_rating: bool = True) -> List[List[int]]:
    data_list = []
    for line in open(path, 'r'):
        user_id, item_id, rating, timestamp = line.strip().split('::')
        user_id = int(user_id)
        item_id = int(item_id)
        rating = int(rating)
        timestamp = int(timestamp)
        if if_rating:
            data_list.append([user_id, item_id, rating])
        else:
            data_list.append([user_id, item_id])
    return data_list


def transfer_list2dict(data: List[List[int]],
                       if_rating: bool = True) -> Dict:
    data_dict = dict()
    if if_rating:
        for line in data:
            user_id, item_id, rating = line
            if data_dict.get(user_id):
                data_dict[user_id][item_id] = rating
            else:
                data_dict[user_id] = {item_id: rating}
    else:
        for line in data:
            user_id, item_id = line
            if data_dict.get(user_id):
                data_dict[user_id].append(item_id)
            else:
                data_dict[user_id] = [item_id]
    return data_dict
