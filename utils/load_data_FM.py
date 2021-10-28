#!/usr/bin/python
# -*- coding: UTF-8 -*-

# @Time    : 2021-10-27 16:14
# @Author  : Eddie Shen
# @Email   : sheneddie@outlook.com
# @File    : load_data_FM.py
# @Software: PyCharm


# %%
# Import packages.
import sys
import numpy as np
import pandas as pd
import torch.utils.data
from torch.utils.data import DataLoader
from typing import Tuple

sys.path.append('../')
try:
    from config.file_path import ml_1m_users_path, ml_1m_ratings_path
    from config.file_path import ml_1m_movies_path
except (ModuleNotFoundError, ImportError):
    raise


# %%
class Preprocess(object):
    def __init__(self, ratio: float = 0.8):
        self.ratio = ratio  # ratio for train set.
        self.users_name = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]
        self.movies_name = ["MovieID", "Title", "Genres"]
        self.ratings_name = ["UserID", "MovieID", "Rating", "Timestamp"]
        # eliminate zip-code, timestamp, title
        self.use_col = ["UserID", "MovieID", "Gender", "Age", "Occupation",
                        'Genres', 'Rating']
        # the dimension of sparse vector (user_id, movie_id)
        self.sparse_n_features = 0
        # the dimension of dense vector
        self.dense_n_features = 0

        self.data = self.load()
        self.train_set, self.test_set = self.split_data()

    def __len__(self):
        return len(self.data)

    def load(self) -> pd.DataFrame:
        users = pd.read_csv(ml_1m_users_path, sep="::", engine='python',
                            names=self.users_name)
        movies = pd.read_csv(ml_1m_movies_path, sep="::", engine='python',
                             names=self.movies_name)
        ratings = pd.read_csv(ml_1m_ratings_path, sep="::", engine='python',
                              names=self.ratings_name)
        data = pd.merge(users, pd.merge(movies, ratings, on="MovieID"),
                        on="UserID")
        # eliminate zip-code, timestamp, title
        data = data[self.use_col]
        return data

    def split_data(self):
        n_train = int(self.__len__() * self.ratio)
        dataset_processed = self.feature_engineering()
        np.random.shuffle(dataset_processed)
        train_set = dataset_processed[:n_train, :]
        test_set = dataset_processed[n_train:, :]
        return train_set, test_set

    def feature_engineering(self):
        user_id = self.user_id_pro()
        item_id = self.item_id_pro()
        gender = self.gender_pro()
        age = self.age_pro()
        occupation = self.occupation_pro()
        genres = self.genres_pro()
        ratings = self.rating_pro()

        user_id = np.expand_dims(user_id, axis=1)
        item_id = np.expand_dims(item_id, axis=1)
        ratings = np.expand_dims(ratings, axis=1)
        dataset = np.concatenate(
            (user_id, item_id, gender, age, occupation, genres, ratings),
            axis=1
        )
        return dataset

    def user_id_pro(self):
        self.sparse_n_features += self.data['UserID'].max()
        user_id = self.data['UserID'].to_numpy()
        user_id = user_id - 1
        return np.float32(user_id)

    def item_id_pro(self):
        self.sparse_n_features += self.data['MovieID'].max()
        item_id = self.data['MovieID'].to_numpy()
        item_id = item_id - 1
        return np.float32(item_id)

    def gender_pro(self):
        self.dense_n_features += 2
        gender = self.data['Gender'].to_numpy()
        gender = np.int32(gender == "M")
        one_hot = np.eye(2, dtype=np.float32)[gender.tolist()]
        return one_hot

    def age_pro(self):
        MAP = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
        self.dense_n_features += len(MAP)
        age = list(map(lambda x: MAP[x], self.data['Age'].tolist()))
        one_hot = np.eye(len(MAP), dtype=np.float32)[age]
        return one_hot

    def occupation_pro(self):
        occupation_max = self.data['Occupation'].max() + 1
        self.dense_n_features += occupation_max
        occupation = self.data['Occupation'].tolist()
        one_hot = np.eye(occupation_max, dtype=np.float32)[occupation]
        return one_hot

    def genres_pro(self):
        MAP = {'Action': 0,
               'Adventure': 1,
               'Animation': 2,
               'Children\'s': 3,
               'Comedy': 4,
               'Crime': 5,
               'Documentary': 6,
               'Drama': 7,
               'Fantasy': 8,
               'Film-Noir': 9,
               'Horror': 10,
               'Musical': 11,
               'Mystery': 12,
               'Romance': 13,
               'Sci-Fi': 14,
               'Thriller': 15,
               'War': 16,
               'Western': 17}
        self.dense_n_features += len(MAP)
        one_hot = np.zeros((self.__len__(), len(MAP)), dtype=np.float32)
        for i, vals in enumerate(self.data['Genres']):
            for val in vals.split('|'):
                j = MAP[val]
                one_hot[i, j] = 1
        return one_hot

    def rating_pro(self):
        ratings = self.data['Rating'].to_numpy()
        return np.float32(ratings)


# %%
class Movie1m(object):
    def __init__(self,
                 preprocess: Preprocess,
                 threshold: float):
        self.train_set = self.rating_threshold(preprocess.train_set, threshold)
        self.test_set = self.rating_threshold(preprocess.test_set, threshold)
        self.sparse_n_features = preprocess.sparse_n_features
        self.dense_n_features = preprocess.dense_n_features

        self.max_user_id = int(max(np.max(self.train_set[:, 0]),
                                   np.max(self.test_set[:, 0])) + 1)
        self.max_item_id = int(max(np.max(self.train_set[:, 1]),
                                   np.max(self.test_set[:, 1])) + 1)

    def rating_threshold(self,
                         dataset: np.ndarray,
                         threshold: float) -> np.ndarray:
        data_res = dataset.copy()
        data_res[:, -1] = np.float32((dataset[:, -1] >= threshold))
        return data_res


class CustomerSet(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        X = self.dataset[index][:-1]
        Y = self.dataset[index][-1]
        return X, Y

    def __len__(self):
        return self.dataset.shape[0]


def get_dataloader(train_set: np.ndarray,
                   test_set: np.ndarray,
                   batch_size: int,
                   cores: int) -> Tuple[DataLoader, DataLoader]:
    train_ds = CustomerSet(train_set)
    test_ds = CustomerSet(test_set)
    train_dl = DataLoader(train_ds,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=cores)
    test_dl = DataLoader(test_ds,
                         batch_size=batch_size,
                         num_workers=cores)
    return train_dl, test_dl
