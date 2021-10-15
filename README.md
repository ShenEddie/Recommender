# 推荐系统实现-说明

## 1.数据集来源

使用的是movielens的1m数据集，网址：[https://files.grouplens.org/datasets/movielens/ml-1m.zip](https://files.grouplens.org/datasets/movielens/ml-1m.zip)

## 2.文件路径设置

在`config/file_path.py`中可设置文件路径：

| 变量               | 说明                             |
| ------------------ | -------------------------------- |
| pwd                | 项目所在文件夹                   |
| ml_1m_ratings_path | movielens数据集中ratings表格路径 |

## 3.模型文件

在`main/`文件夹下放置了项目的模型文件，可直接运行：

| 文件      | 说明               |
| --------- | ------------------ |
| ItemCF.py | 基于物品的邻域算法 |
| LFM.py    | 隐语义模型         |
| UserCF.py | 基于用户的邻域算法 |
