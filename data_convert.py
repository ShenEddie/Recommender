#!/usr/bin/python
# -*- coding: UTF-8 -*-

# @Time    : 2021-09-15 15:46
# @Author  : Eddie Shen
# @Email   : sheneddie@outlook.com
# @File    : data_convert.py
# @Software: PyCharm

# %%
# Import packages.
import pandas as pd

# %%
# Read data.
df: pd.DataFrame = pd.read_csv('./data/movies/ratings.csv')
df.columns = ['user_id', 'item_id', 'rating', 'timestamp']

# %%
n_user = df['user_id'].nunique()
n_item = df['item_id'].nunique()

# %%
# Remap item_id.
id_map = pd.DataFrame({
    'old_id': df['item_id'].unique(),
    'new_id': list(range(1, n_item + 1))
})
ratings: pd.DataFrame = pd.merge(left=df, right=id_map, how='left', left_on='item_id', right_on='old_id')
ratings = ratings[['user_id', 'new_id', 'rating']]
ratings.rename(columns={'new_id': 'item_id'}, inplace=True)
ratings.head()

# %%
# Save to pickle.
ratings.to_pickle('./data/movies.pkl')
