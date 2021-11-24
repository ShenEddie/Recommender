# 推荐系统实现-说明

## 1.数据集来源

使用的是movielens的1m数据集，网址：[https://files.grouplens.org/datasets/movielens/ml-1m.zip](https://files.grouplens.org/datasets/movielens/ml-1m.zip)

## 2.文件路径设置

在`config/file_path.py`中可设置文件路径：

| 变量                   | 说明                                         |
| ---------------------- | -------------------------------------------- |
| pwd                    | 项目所在文件夹                               |
| ml_1m_ratings_path     | movielens数据集中用户-电影评分表格路径       |
| ml_1m_users_path       | movielens数据集中用户属性表格路径            |
| ml_1m_movies_path      | movielens数据集中电影属性表格路径            |
| user_cf_W_path         | UserCF模型权重参数保存路径                   |
| user_cf_W_iif_path     | UserIIF模型权重参数保存路径                  |
| item_cf_W_path         | ItemCF模型权重参数保存路径                   |
| item_cf_W_iuf_path     | ItemIUF模型权重参数保存路径                  |
| item_cf_W_penalty_path | 带惩罚项的ItemCF模型权重参数保存路径         |
| item_cf_W_normed_path  | 对数据进行标准化的ItemCF模型权重参数保存路径 |
| lfm_params_path        | LFM模型的权重参数保存路径                    |

## 3.模型文件

在`main/`文件夹下放置了项目的模型文件，可直接运行：

| 文件      | 说明               |
| --------- | ------------------ |
| ItemCF.py | 基于物品的邻域算法 |
| LFM.py    | 隐语义模型         |
| UserCF.py | 基于用户的邻域算法 |
| FM.py     | 因子分解机(FM)算法 |

## 4.工具文件

在`utils/`文件夹下放置了项目的部分工具类文件：

| 文件            | 说明                                                     |
| --------------- | -------------------------------------------------------- |
| load_data.py    | UserCF、ItemCF、LFM模型的数据读取相关函数                |
| load_data_FM.py | FM模型的数据读取、特征工程相关的类和函数                 |
| decorate_FM.py  | FM模型的函数装饰器，主要用于显示模型每次训练的时间等信息 |
| session_FM.py   | FM模型的session定义文件                                  |

