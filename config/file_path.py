#!/usr/bin/python
# -*- coding: UTF-8 -*-

# @Time    : 2021-09-24 10:58
# @Author  : Eddie Shen
# @Email   : sheneddie@outlook.com
# @File    : file_path.py
# @Software: PyCharm

import os.path

pwd = r'D:\Eddie\Documents\BOSC\RecommendSystem'
ml_1m_ratings_path = os.path.join(pwd, r'data/ml-1m/ratings.dat')
user_cf_W_path = os.path.join(pwd, r'params/W_UserCF.pkl')
user_cf_W_iif_path = os.path.join(pwd, r'params/W_iif_UserCF.pkl')
lfm_params_path = os.path.join(pwd, r'params/P_Q_lfm.pkl')
