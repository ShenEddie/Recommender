#!/usr/bin/python
# -*- coding: UTF-8 -*-

# @Time    : 2021-10-27 11:05
# @Author  : Eddie Shen
# @Email   : sheneddie@outlook.com
# @File    : FM.py
# @Software: PyCharm


# %%
# Import packages.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


# %%
class FactorMachine(nn.Module):
    def __init__(self,
                 sparse_n_features: int,
                 dense_n_features: int,
                 n_classes: int,
                 n_factors: int,
                 batch_size: int,
                 decay):
        super(FactorMachine, self).__init__()
        self.sparse_n_features = sparse_n_features
        self.dense_n_features = dense_n_features
        self.n_classes = n_classes
        self.n_factors = n_factors
        self.batch_size = batch_size
        self.decay = decay

        self.n_features = self.sparse_n_features + self.dense_n_features
        self.w0, self.w1, self.w2 = self.init_weight()

    def init_weight(self) -> Tuple[Tensor, Tensor, Tensor]:
        init = nn.init.normal_
        w_0 = nn.Parameter(nn.init.constant_(torch.empty(1, ), 0.0))
        w_1 = nn.Parameter(init(torch.empty((self.n_features, 1)), std=0.01))
        w_2 = nn.Parameter(
            init(torch.empty((self.n_features, self.n_factors)), std=0.01)
        )
        return w_0, w_1, w_2

    def fold_mm(self, user_item_sparse, other_features, weight) -> Tensor:
        res = torch.sparse.mm(user_item_sparse,
                              weight[:self.sparse_n_features, :]) + \
              torch.matmul(other_features, weight[self.sparse_n_features:, :])
        return res

    def forward(self, user_item_sparse, other_features) -> Tensor:
        linear_term = \
            self.w0 + self.fold_mm(user_item_sparse, other_features, self.w1)
        hybird_term = \
            self.fold_mm(user_item_sparse, other_features, self.w2) ** 2 - \
            self.fold_mm(user_item_sparse ** 2, other_features ** 2,
                         self.w2 ** 2)

        linear_term = torch.squeeze(linear_term)
        y_pred = linear_term + 0.5 * torch.sum(hybird_term, dim=1)
        return nn.Sigmoid()(y_pred)

    def loss_func(self, y_truth, y_pred):
        criterion = nn.BCELoss()
        cross_entropy = criterion(y_pred, y_truth)
        l2_norm = 0.5 * (torch.norm(self.w0) ** 2 + torch.norm(self.w1) ** 2 +
                         torch.norm(self.w2) ** 2)
        l2_norm = l2_norm * self.decay / self.batch_size
        loss = cross_entropy + l2_norm
        return loss, cross_entropy, l2_norm
