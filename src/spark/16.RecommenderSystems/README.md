# 特征工程部分
代码位于Features 类
因为现实系统的rating 值，非常稀疏，为了节省存储空间和提升效率，在特征存储结构上，需要进行一些改动。主要思路如下：

## 将rating 转为libsvm的方式存储
用户对物品打分的数据，针对单个用户而言，对不同的物品的打分是很稀疏的。可以用libsvm格式来进行存储。
```
例如： 输入为
1::661::3::978302109
1::914::3::978301968
转化之后结果为
1 661:3 914:3
```

## 将rating 转为<id, features> 格式的DataFrame
id 为String 
features 为 SparseVector

# ItemCF
代码位于ItemCF 类
## 相似度计算
实现了两种方式，Jaccard 相似度 和 余弦相似度

## Jaccard 
使用BitSet 存储每个用户的对该Item 是否有Ratting 的情况。

## 余弦相似度
使用自带API实现


## 基于Item 相似度 推荐单个物品
选取该物品和其他物品的相似度向量。使用特征向量和相似度向量点乘即可。


## 基于Item 相似度  推荐topK 的物品
挖坑

# UserCF
挖坑
## 根据topN的相似用户推荐
 