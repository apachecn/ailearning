# coding: utf-8

"""
TF-IDF 模型：
主要思想是，如果某个词或短语在一篇文章中出现的频率TF(Term Frequency，词频)，词频高，
    并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类。

公式：
* TF-IDF = TF * IDF
* TF(t) = (词t在文档中出现的总次数) / (文档的词总数)
* IDF = log_e(总文档数/词t出现的文档数)
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# 语料
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]

"""
计算方式1: 通过手动的进行计算词频
"""
# 将文本中的词语转换为词频矩阵
cv = CountVectorizer(min_df=1)
# 计算个词语出现的次数
cv_fit = cv.fit_transform(corpus)
print("bag of words（BOW） 词袋模型")
print(cv_fit.toarray())

print("\n", "---" * 10, "\n")

# 计算 IF-IDF的值
transformer = TfidfTransformer()
# 将词频矩阵 cv_fit 统计成 TF-IDF 值
tfidf = transformer.fit_transform(cv_fit)
# 查看数据结构 tfidf[i][j] 表示i类文本中 tf-idf 权重
print("TF-IDF 模型")
print(tfidf.toarray())


print("\n", "---" * 10, "\n")
"""
计算方式2: 直接通过 文本词料 来计算文本中 tf-idf 权重
"""
# 计算 IF-IDF的值
transformer = TfidfVectorizer()
# 将词频矩阵 cv 统计成 TF-IDF 值
tfidf = transformer.fit_transform(corpus)
# 查看数据结构 tfidf[i][j] 表示i类文本中的 tf-idf 权重
print("TF-IDF 模型")
print(tfidf.toarray())
