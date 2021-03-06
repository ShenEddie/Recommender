#!/usr/bin/python
# -*- coding: UTF-8 -*-

# @Time    : 2021-09-24 10:58
# @Author  : Eddie Shen
# @Email   : sheneddie@outlook.com
# @File    : file_path.py
# @Software: PyCharm

import os.path

pwd = 'D:\\datutu\\BOSC\\Recommender'
ml_1m_ratings_path = os.path.join(pwd, r'data/ml-1m/ratings.dat')
ml_1m_users_path = os.path.join(pwd, r'data/ml-1m/users.dat')
ml_1m_movies_path = os.path.join(pwd, r'data/ml-1m/movies.dat')
user_cf_W_path = os.path.join(pwd, r'params/W_UserCF.pkl')
user_cf_W_iif_path = os.path.join(pwd, r'params/W_iif_UserCF.pkl')
item_cf_W_path = os.path.join(pwd, r'params/W_ItemCF.pkl')
item_cf_W_iuf_path = os.path.join(pwd, r'params/W_iuf_ItemCF.pkl')
item_cf_W_penalty_path = os.path.join(pwd, r'params/W_penalty_ItemCF.pkl')
item_cf_W_normed_path = os.path.join(pwd, r'params/W_normed_ItemCF.pkl')
lfm_params_path = os.path.join(pwd, r'params/P_Q_lfm.pkl')
