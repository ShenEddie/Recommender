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
except ModuleNotFoundError:
    raise


def load_m1_1m(path: str = ml_1m_ratings_path,
               type_res: str = 'list',
               if_rating: bool = False) -> Union[list, dict]:
    data_dict = {}
    data_list = []
    for line in open(path, 'r'):
        user_id, item_id, rating, timestamp = line.strip().split('::')
        user_id = int(user_id)
        item_id = int(item_id)
        rating = int(rating)
        timestamp = int(timestamp)

        if type_res == 'dict':
            if if_rating:
                if data_dict.get(user_id):
                    data_dict[user_id][item_id] = rating
                else:
                    data_dict[user_id] = {item_id: rating}
            else:
                if data_dict.get(user_id):
                    data_dict[user_id].append(item_id)
                else:
                    data_dict[user_id] = [item_id]
        elif type_res == 'list':
            if if_rating:
                data_list.append([user_id, item_id, rating])
            else:
                data_list.append([user_id, item_id])

    # Return result.
    if type_res == 'dict':
        return data_dict
    elif type_res == 'list':
        return data_list
    else:
        raise RuntimeError('Wrong type_res')
