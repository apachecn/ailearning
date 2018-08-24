# coding: utf-8

# 第三方分词工具
import jieba

# 分词模式
seg = jieba.cut("这是一本关于信息检索的书", cut_all=True)  # cut_all=True，全模式
print("\n全模式分词: \n", "/ ".join(seg))

seg = jieba.cut("这是一本关于信息检索的书", cut_all=False)  # cut_all=False，精确模式
print("\n精确模式分词: \n", "/ ".join(seg))

seg = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print("\n测试精确模式: \n", ", ".join(seg))


# 搜索引擎模式
seg = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")
print("\n测试搜索引擎模式: \n", ", ".join(seg))


# 添加自定义词典
jieba.load_userdict("src/py3.x/NLP/5.jieba-model/userdic.txt")
seg = jieba.cut("这是一本关于八一双鹿信息检索的书")
print("\n测试自定义字典: \n", "/ ".join(seg))
